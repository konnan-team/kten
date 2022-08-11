package eu.redbean.kten.opencl.tensor.platform

import eu.redbean.kten.api.tensor.platform.DeviceType
import eu.redbean.kten.api.tensor.platform.PlatformInfo
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.SynchronizationLevel
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.SynchronizationLevel.OFF_DEVICE
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.SynchronizationLevel.ON_DEVICE
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor
import org.jocl.*
import org.jocl.blast.CLBlast
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class OCLEnvironment(deviceIndex: Int) {

    val context: cl_context = createContext(deviceIndex)
    val commandQueue: cl_command_queue = createCommandQueue(deviceIndex, context)
    val kernelStore: OCLKernelStore by lazy {
        OCLKernelStore(
            context,
            commandQueue,
            oclPlatformInfos[deviceIndex]!!.platformSpecificInfo as OCLPlatformSpecInfo
        )
    }

    private val threadLocalInstanceCollector: ThreadLocal<(OCLRawTensor) -> Unit> = ThreadLocal.withInitial({ {} })

    internal var instanceCollector: (OCLRawTensor) -> Unit
        get() = threadLocalInstanceCollector.get()
        set(value) = threadLocalInstanceCollector.set(value)

    private val memObjectReusePool = ConcurrentHashMap<Int, ConcurrentLinkedQueue<OCLMemoryObject>>()

    private val estReusePoolMemUsage = AtomicLong(0L)

    private val activeMemObjects = ConcurrentLinkedQueue<OCLMemoryObject>()

    private val estActiveMemUsage = AtomicLong(0L)

    private val stopTheWorld = AtomicBoolean()

    private val maxMemory: Long

    private val maxAllowedEstimatedMemUsage: Long

    private var memoryManagementWarningPrinted = false

    init {
        maxMemory = oclPlatformInfos[deviceIndex]!!.availableMemory
        maxAllowedEstimatedMemUsage = (maxMemory * PlatformProvider.memoryUsageScaleHint).toLong()
        val cleanupThread = Thread {
            while (true) {
                memoryCleanup()
                Thread.sleep(100)
            }
        }
        cleanupThread.isDaemon = true
        cleanupThread.start()
    }

    fun memoryCleanup() {
        if (stopTheWorld.get().not() && estActiveMemUsage.get() > maxAllowedEstimatedMemUsage) {
            stopTheWorld.set(true)
            memObjectReusePool.forEachValue(10L) {
                while (it.size > 0) {
                    val memObject = it.poll()
                    if (memObject?.isReusable() == true) { // just to be safe
                        estReusePoolMemUsage.addAndGet(-(memObject.memSize))
                        memObject.release()
                    }
                }
            }
            if (PlatformProvider.useJVMMemoryAsCache.not() && estActiveMemUsage.get() > maxAllowedEstimatedMemUsage) {
                activeMemObjects
                    .filter { it.manageUnusedDeviceMemory() }
                    .forEach {
                        activeMemObjects.remove(it)
                        estActiveMemUsage.addAndGet(-it.memSize)
                    }
            }
            stopTheWorld.set(false)
        }
    }

    fun getPlatformMetrics(): Map<String, Float> {
        val estMemoryUsageInMB = estActiveMemUsage.get() / 1024f / 1024f
        val estReuseMemoryUsageInMB = estReusePoolMemUsage.get() / 1024f / 1024f
        return mapOf(
            "Estimated memory usage (MB)" to estMemoryUsageInMB,
            "Estimated reuse pool memory usage (MB)" to estReuseMemoryUsageInMB
        )
    }

    private fun getMemObjectFromPool(size: Int): OCLMemoryObject? {
        while (stopTheWorld.get()) {
            Thread.sleep(1)
        }
        var memoryObject: OCLMemoryObject?
        do {
            memoryObject = memObjectReusePool.get(size)?.poll()
        } while (memoryObject != null && memoryObject.isReusable().not())
        return memoryObject
    }

    fun registerActiveMemObject(memoryObject: OCLMemoryObject) {
        activeMemObjects.offer(memoryObject)
        estActiveMemUsage.addAndGet(memoryObject.memSize)
    }

    fun removeInactiveMemObject(memoryObject: OCLMemoryObject) {
        activeMemObjects.remove(memoryObject)
        estActiveMemUsage.addAndGet(-memoryObject.memSize)
    }

    /**
     * If system memory caching is enabled, it tries to clean up the reuse pool first, if memory limit is hit. If that doesn't help
     * it tries to clean up space on the device for the required memory allocation, by transferring memory objects' data from the device
     * memory to the JVM Heap.
     *
     * This is called for every memory object allocation. It runs in an STW event, to avoid problems in the cleanup thread.
     *
     * The first time it moves memory objects from device to system memory it prints a warning, that this will cause performance issues.
     */
    private fun manageActiveMemory(requiredMemory: Long) {
        if (PlatformProvider.useJVMMemoryAsCache.not())
            return

        if (estActiveMemUsage.get() >= maxAllowedEstimatedMemUsage) {
            memoryCleanup()
        }

        if (estActiveMemUsage.get() >= maxAllowedEstimatedMemUsage) {
            stopTheWorld.set(true)

            if (memoryManagementWarningPrinted.not()) {
                println("WARNING: Active device memory limit exceeded. Usage of system memory cache is enabled, " +
                        "which means tensors will be moved to the JVM Heap.")
                println("This may cause significant slowdown of the calculations, and if not enough heap space is provided, " +
                        "even can lead to OutOfMemory error.")
                memoryManagementWarningPrinted = true
            }

            var freeableSize = 0L
            val lastAccessed = activeMemObjects.sortedBy { it.lastAccess }
            val moveToSystemMem = mutableListOf<OCLMemoryObject>()
            for (memObj in lastAccessed) {
                moveToSystemMem.add(memObj)
                freeableSize += memObj.memSize
                if (freeableSize >= requiredMemory)
                    break
            }
            moveToSystemMem.forEach {
                activeMemObjects.remove(it)
                it.manageUnusedDeviceMemory(true)
                estActiveMemUsage.addAndGet(-it.memSize)
            }
            stopTheWorld.set(false)
        }
    }

    fun memoryObjectOf(array: FloatArray, synchronization: SynchronizationLevel = OFF_DEVICE): OCLMemoryObject {
        val memoryObjectFromPool = getMemObjectFromPool(array.size)
        if (memoryObjectFromPool?.isReusable() == true) {
            memoryObjectFromPool.reuseWithJvmArray(array, synchronization)
            estReusePoolMemUsage.addAndGet(-memoryObjectFromPool.memSize)
            return memoryObjectFromPool
        }

        manageActiveMemory(array.size.toLong() * Sizeof.cl_float)

        return OCLMemoryObject(array, this, synchronization)
    }

    fun memoryObject(size: Int, synchronization: SynchronizationLevel = ON_DEVICE): OCLMemoryObject {
        val memoryObjectFromPool = getMemObjectFromPool(size)
        if (memoryObjectFromPool?.isReusable() == true) {
            memoryObjectFromPool.reuseWithoutJvmBacking(synchronization)
            estReusePoolMemUsage.addAndGet(-memoryObjectFromPool.memSize)
            return memoryObjectFromPool
        }

        manageActiveMemory(size.toLong() * Sizeof.cl_float)

        return OCLMemoryObject(size, this, synchronization)
    }

    fun mayReuseOrRelease(memoryObject: OCLMemoryObject) {
        memoryObject.makeReusable()
        if (memoryObject.isReusable()) {
            val reuseQueue = memObjectReusePool.computeIfAbsent(memoryObject.size) {
                ConcurrentLinkedQueue<OCLMemoryObject>()
            }
            estReusePoolMemUsage.addAndGet(memoryObject.memSize)
            reuseQueue.offer(memoryObject)
        }
    }

    companion object {
        private val contextProperties = cl_context_properties()
        private lateinit var devices: Array<cl_device_id?>
        private lateinit var oclPlatformInfos: Map<Int, PlatformInfo>

        fun getDevices(): Map<Int, PlatformInfo> {
            CL.setExceptionsEnabled(true)
            CLBlast.setExceptionsEnabled(true)

            val platformArray = IntArray(1)
            CL.clGetPlatformIDs(0, null, platformArray)
            val numberOfPlatforms = platformArray[0]

            val platforms = Array<cl_platform_id?>(numberOfPlatforms) { null }
            CL.clGetPlatformIDs(platforms.size, platforms, null)
            val platform = platforms[0]

            contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM.toLong(), platform)

            val numberOfDevicesArray = IntArray(1)
            CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, 0, null, numberOfDevicesArray)
            val numberOfDevices = numberOfDevicesArray[0]

            devices = Array<cl_device_id?>(numberOfDevices) { null }
            CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, numberOfDevices, devices, null)

            oclPlatformInfos = devices.mapIndexed(::mapIndexToPlatformInfo).toMap()
            return oclPlatformInfos
        }

        private fun mapIndexToPlatformInfo(index: Int, deviceId: cl_device_id?): Pair<Int, PlatformInfo> {
            val device = deviceId!!

            val size = LongArray(1)
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_NAME, 0, null, size)
            val buffer = ByteArray(size[0].toInt())
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_NAME, buffer.size.toLong(), Pointer.to(buffer), null)
            val deviceName = String(buffer, 0, buffer.size - 1)

            val typeSize = LongArray(1)
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_TYPE, 0, null, typeSize)
            val typeBuffer = LongArray(typeSize[0].toInt())
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_TYPE, typeBuffer.size.toLong(), Pointer.to(typeBuffer), null)
            val deviceType = when (typeBuffer[0]) {
                CL.CL_DEVICE_TYPE_CPU -> DeviceType.CPU
                CL.CL_DEVICE_TYPE_GPU -> DeviceType.GPU
                CL.CL_DEVICE_TYPE_ACCELERATOR -> DeviceType.OTHER //TODO maybe these should be added to the DeviceType?
                CL.CL_DEVICE_TYPE_DEFAULT -> DeviceType.OTHER
                CL.CL_DEVICE_TYPE_CUSTOM -> DeviceType.OTHER
                else -> DeviceType.OTHER
            }

            val memSize = LongArray(1)
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_GLOBAL_MEM_SIZE, 0, null, memSize)
            val memBuffer = LongArray(memSize[0].toInt())
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_GLOBAL_MEM_SIZE, memBuffer.size.toLong(), Pointer.to(memBuffer), null)
            val maxMemory = memBuffer[0]

            val maxWorkGroupSize = LongArray(1)
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, null, maxWorkGroupSize)
            val mwgSizeBuffer = LongArray(maxWorkGroupSize[0].toInt())
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_MAX_WORK_GROUP_SIZE, mwgSizeBuffer.size.toLong(), Pointer.to(mwgSizeBuffer), null)
            val maxWorkGroup = mwgSizeBuffer[0]

            val maxWorkItemSize = LongArray(1)
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, null, maxWorkItemSize)
            val mwiSizeBuffer = LongArray(maxWorkItemSize[0].toInt())
            CL.clGetDeviceInfo(device, CL.CL_DEVICE_MAX_WORK_ITEM_SIZES, mwiSizeBuffer.size.toLong(), Pointer.to(mwiSizeBuffer), null)
            val maxWorkItems = mwiSizeBuffer.toList()

            val platformInfo = PlatformInfo(
                "OpenCL", deviceType, maxMemory, "OpenCL - $index - $deviceName",
                OCLPlatformSpecInfo(maxWorkGroup, maxWorkItems, device)
            )

            return index to platformInfo
        }

        fun createContext(deviceIndex: Int): cl_context {
            val device = devices[deviceIndex]

            return CL.clCreateContext(
                contextProperties,
                1,
                Array(1) { device },
                null, null, null
            )
        }

        fun createCommandQueue(deviceIndex: Int, context: cl_context): cl_command_queue {
            val device = devices[deviceIndex]
            @Suppress("DEPRECATION") //new API is not supported on macos
            return CL.clCreateCommandQueue(
                context,
                device,
                0,
                null
            )
        }
    }

}
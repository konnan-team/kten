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

    init {
        val maxMemory = oclPlatformInfos[deviceIndex]!!.availableMemory
        val maxAllowedEstimatedMemUsage = (maxMemory * PlatformProvider.memoryUsageScaleHint).toLong()
        val cleanupThread = Thread {
            while (true) {
                if (estReusePoolMemUsage.get() > (maxAllowedEstimatedMemUsage * .75).toLong()) {
                    memObjectReusePool.forEachValue(10L) {
                        if (it.size > 5)
                            while (it.size > 5 && estReusePoolMemUsage.get() > (maxAllowedEstimatedMemUsage * .75).toLong()) {
                                val memObject = it.poll()
                                if (memObject?.isReusable() == true) { // just to be safe
                                    estReusePoolMemUsage.addAndGet(-(memObject.memSize))
                                    memObject.release()
                                }
                            }
                    }
                }
                if (estActiveMemUsage.get() > maxAllowedEstimatedMemUsage) {
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
                    if (estActiveMemUsage.get() > maxAllowedEstimatedMemUsage) {
                        activeMemObjects
                            .filter { it.manageUnusedDeviceMemory() }
                            .forEach {
                                activeMemObjects.remove(it)
                                estActiveMemUsage.addAndGet(-it.memSize)
                            }
                    }

                    while (estActiveMemUsage.get() > maxMemory * 0.9) {
                        val memObject = activeMemObjects.poll()
                        memObject.manageUnusedDeviceMemory(true)
                        estActiveMemUsage.addAndGet(-memObject.memSize)
                    }
                    stopTheWorld.set(false)
                }

                Thread.sleep(100)
            }
        }
        cleanupThread.isDaemon = true
        cleanupThread.start()
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

    fun memoryObjectOf(array: FloatArray, synchronization: SynchronizationLevel = OFF_DEVICE): OCLMemoryObject {
        val memoryObjectFromPool = getMemObjectFromPool(array.size)
        if (memoryObjectFromPool?.isReusable() == true) {
            memoryObjectFromPool.reuseWithJvmArray(array, synchronization)
            estReusePoolMemUsage.addAndGet(-memoryObjectFromPool.memSize)
            return memoryObjectFromPool
        }

        return OCLMemoryObject(array, this, synchronization)
    }

    fun memoryObject(size: Int, synchronization: SynchronizationLevel = ON_DEVICE): OCLMemoryObject {
        val memoryObjectFromPool = getMemObjectFromPool(size)
        if (memoryObjectFromPool?.isReusable() == true) {
            memoryObjectFromPool.reuseWithoutJvmBacking(synchronization)
            estReusePoolMemUsage.addAndGet(-memoryObjectFromPool.memSize)
            return memoryObjectFromPool
        }

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
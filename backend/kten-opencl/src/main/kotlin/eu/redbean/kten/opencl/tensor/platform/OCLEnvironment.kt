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
import java.util.concurrent.atomic.AtomicLong

class OCLEnvironment(deviceIndex: Int) {

    val context: cl_context = createContext(deviceIndex)
    val commandQueue: cl_command_queue = createCommandQueue(deviceIndex, context)
    val kernelStore: OCLKernelStore by lazy { OCLKernelStore(context, commandQueue) }

    private val threadLocalInstanceCollector: ThreadLocal<(OCLRawTensor) -> Unit> = ThreadLocal.withInitial({ {} })

    internal var instanceCollector: (OCLRawTensor) -> Unit
        get() = threadLocalInstanceCollector.get()
        set(value) = threadLocalInstanceCollector.set(value)

    private val memObjectReusePool = ConcurrentHashMap<Int, ConcurrentLinkedQueue<OCLMemoryObject>>()

    private val estReusePoolMemUsage = AtomicLong(0L)

    init {
        val maxMemory = oclPlatformInfos[deviceIndex]!!.availableMemory
        val maxAllowedReuseSize = (maxMemory * PlatformProvider.memoryUsageScaleHint).toLong()
        Thread {
            while (true) {
                if (estReusePoolMemUsage.get() > maxAllowedReuseSize) {
                    memObjectReusePool.forEachValue(10L) {
                        if (it.size > 2)
                            OCLPlatformInitializer.releaseExecutor.execute {
                                while (it.size > 2) {
                                    val memObject = it.poll()
                                    estReusePoolMemUsage.addAndGet(-(memObject?.memSize ?: 0L))
                                    if (memObject?.isReusable() == true) // just to be safe
                                        memObject.release()
                                }
                            }
                    }
                }
                Thread.sleep(200)
            }
        }.start()
    }

    fun memoryObjectOf(array: FloatArray, synchronization: SynchronizationLevel = OFF_DEVICE): OCLMemoryObject {
        val memoryObjectFromPool = memObjectReusePool.get(array.size)?.poll()
        if (memoryObjectFromPool?.isReusable() == true) {
            memoryObjectFromPool.reuseWithJvmArray(array, synchronization)
            estReusePoolMemUsage.addAndGet(-memoryObjectFromPool.memSize)
            return memoryObjectFromPool
        }

        return OCLMemoryObject(array, this, synchronization)
    }

    fun memoryObject(size: Int, synchronization: SynchronizationLevel = ON_DEVICE): OCLMemoryObject {
        val memoryObjectFromPool = memObjectReusePool.get(size)?.poll()
        if (memoryObjectFromPool?.isReusable() == true) {
            memoryObjectFromPool.reuseWithoutJvmBacking(synchronization)
            estReusePoolMemUsage.addAndGet(-memoryObjectFromPool.memSize)
            return memoryObjectFromPool
        }

        return OCLMemoryObject(size, this, synchronization)
    }

    fun mayReuseOrRelease(memoryObject: OCLMemoryObject) {
        val reuseQueue = memObjectReusePool.computeIfAbsent(memoryObject.size) {
            ConcurrentLinkedQueue<OCLMemoryObject>()
        }
        memoryObject.makeReusable()
        estReusePoolMemUsage.addAndGet(memoryObject.memSize)
        reuseQueue.offer(memoryObject)
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

            val platformInfo = PlatformInfo(
                "OpenCL", deviceType, maxMemory, "OpenCL - $index - $deviceName"
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
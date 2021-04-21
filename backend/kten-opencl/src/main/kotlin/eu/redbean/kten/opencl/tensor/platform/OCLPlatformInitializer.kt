package eu.redbean.kten.opencl.tensor.platform

import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.platform.PlatformInitializer
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.jvm.tensor.operations.JVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLRawTensor
import java.lang.ref.Cleaner
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.Executors

object OCLPlatformInitializer: PlatformInitializer {

    override val platformKeys: List<String>

    val cleaner = Cleaner.create()

    val releaseExecutor = Executors.newFixedThreadPool(2)


    init {
        val devices = OCLEnvironment.getDevices()
        val tensorOps = devices.map { (deviceIndex, deviseName) ->
            val platformKey = "OpenCL - $deviceIndex - $deviseName"
            platformKey to OCLTensorOperations(platformKey) { OCLEnvironment(deviceIndex) }
        }
        tensorOps.forEach { (platformKey, tensorOp) ->
            PlatformProvider.register(platformKey, tensorOp as TensorOperations<AbstractRawTensor<Any>>)
            PlatformProvider.registerPlatformTransformer(JVMTensorOperations.platformKey to platformKey) {
                val jvmRawTensor = it as JVMRawTensor
                val oclRawTensor = tensorOp.createRaw(jvmRawTensor.shape) { jvmRawTensor.storeReference[it] }
                oclRawTensor.storeReference.writeToDevice()
                oclRawTensor as AbstractRawTensor<Any>
            }
            PlatformProvider.registerPlatformTransformer(platformKey to JVMTensorOperations.platformKey) {
                val oclRawTensor = it as OCLRawTensor
                oclRawTensor.storeReference.readToArray()
                JVMTensorOperations.createRaw(oclRawTensor.shape) { oclRawTensor.storeReference[it] } as AbstractRawTensor<Any>
            }
        }
        platformKeys = tensorOps.map { it.first }
    }

}
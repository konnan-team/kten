package eu.redbean.kten.jvm.tensor.operations

import eu.redbean.kten.api.autograd.tensor.Variable
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.TensorOperationsGarbageCollector
import eu.redbean.kten.api.tensor.platform.DeviceType
import eu.redbean.kten.api.tensor.platform.PlatformInfo
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import java.util.concurrent.atomic.AtomicLong

object MemLeakDetectingJVMTensorOperations: AbstractJVMTensorOperations() {

    override val platformKey = "MemLeakDetectJVM"

    internal var rawMemUsage = AtomicLong(0L)

    internal var instanceCollector: (JVMRawTensor) -> Unit = {}

    private var alreadyInGC = false

    val platformInfo = PlatformInfo(
        "JVM - Testing", DeviceType.JVM, Runtime.getRuntime().maxMemory(), platformKey
    )

    init {
        PlatformProvider.register(platformKey, this as TensorOperations<AbstractRawTensor<Any>>, platformInfo)

        PlatformProvider.registerPlatformTransformer(JVMTensorOperations.platformKey to this.platformKey) {
            JVMRawTensor(it.shape, it.storeReference as FloatArray, this.platformKey) as AbstractRawTensor<Any>
        }
        PlatformProvider.registerPlatformTransformer(this.platformKey to JVMTensorOperations.platformKey) {
            JVMRawTensor(it.shape, it.storeReference as FloatArray, JVMTensorOperations.platformKey) as AbstractRawTensor<Any>
        }
    }

    override fun release(vararg rawTensors: AbstractRawTensor<Any>) {
        release(rawTensors.map { it as JVMRawTensor })
    }

    fun release(rawTensors: Iterable<JVMRawTensor>) {
        rawTensors.forEach {
            val tensorSize = it.storeReference.size * 4L
            if (it.release())
                rawMemUsage.addAndGet(-tensorSize)
        }
    }

    override fun garbageCollector(): TensorOperationsGarbageCollector {
        if (alreadyInGC) //overlapping case
            return TensorOperationsGarbageCollector {  }

        val instances = mutableListOf<JVMRawTensor>()
        instanceCollector = {
            if (Thread.currentThread().threadGroup.name != "prefetch") //TODO make it configurable
                instances.add(it)
        }
        alreadyInGC = true
        return TensorOperationsGarbageCollector {
            instanceCollector = {}
            alreadyInGC = false
            release(instances)
            instances.clear()
        }
    }

    fun referenceStat() {
        println("Approx. mem usage: ${rawMemUsage.get() / 1024}kB")
    }

    override fun incrementRef(rawTensor: JVMRawTensor) {
        rawTensor.incrementRef()
    }

}
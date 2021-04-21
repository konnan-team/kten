package eu.redbean.kten.jvm.tensor.operations

import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

object JVMTensorOperations: AbstractJVMTensorOperations() {

    override val platformKey: String = "JVM"

    init {
        PlatformProvider.registerAsDefault(platformKey, this as TensorOperations<AbstractRawTensor<Any>>)
    }

}
package eu.redbean.kten.jvm.tensor.platform

import eu.redbean.kten.api.tensor.platform.PlatformInitializer
import eu.redbean.kten.jvm.tensor.operations.JVMTensorOperations
import eu.redbean.kten.jvm.tensor.operations.MemLeakDetectingJVMTensorOperations

@Suppress("unused")
object JVMPlatformInitializer: PlatformInitializer {

    override val platformKeys = listOf(
        JVMTensorOperations.platformKey,
        MemLeakDetectingJVMTensorOperations.platformKey
    )

}
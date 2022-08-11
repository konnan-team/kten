package eu.redbean.kten.tensor.nn

import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.TensorOperationsGarbageCollector
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.jvm.tensor.operations.MemLeakDetectingJVMTensorOperations
import eu.redbean.kten.tensor.tests.nn.PoolingTestsBase
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.TestInstance

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MemoryLeakDetectingPoolingTests: PoolingTestsBase() {

    private lateinit var gc: TensorOperationsGarbageCollector

    @BeforeAll
    fun setupDefaultOpsAndGC() {
        PlatformProvider.registerAsDefault(
            MemLeakDetectingJVMTensorOperations.platformKey,
            MemLeakDetectingJVMTensorOperations as TensorOperations<AbstractRawTensor<Any>>,
            MemLeakDetectingJVMTensorOperations.platformInfo
        )
        gc = MemLeakDetectingJVMTensorOperations.garbageCollector()
    }

    @AfterAll
    fun tearDownGCAndPrintMemUsage() {
        gc.close()
        MemLeakDetectingJVMTensorOperations.referenceStat()
    }

}
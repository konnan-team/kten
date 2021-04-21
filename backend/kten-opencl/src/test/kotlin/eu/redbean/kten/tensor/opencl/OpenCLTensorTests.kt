package eu.redbean.kten.tensor.opencl

import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.TensorOperationsGarbageCollector
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.tensor.tests.TensorTestsBase
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.TestInstance

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class OpenCLTensorTests: TensorTestsBase() {

    private lateinit var gc: TensorOperationsGarbageCollector

    private val platformKey = "OpenCL - 0 - Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz"//"OpenCL - 2 - AMD Radeon Pro 455 Compute Engine"

    @BeforeAll
    fun setupDefaultOps() {
        PlatformProvider.registerAsDefault(
            platformKey,
            PlatformProvider.tensorOperations(platformKey) as TensorOperations<AbstractRawTensor<Any>>
        )
    }

    @BeforeEach
    fun setupGc() {
        gc = PlatformProvider.tensorOperations(platformKey).garbageCollector()
    }

    @AfterEach
    fun tearDownGc() {
        gc.close()
    }

}
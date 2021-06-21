package eu.redbean.kten.tensor.opencl

import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.TensorOperationsGarbageCollector
import eu.redbean.kten.api.tensor.platform.DeviceType
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

    private lateinit var platformKey: String

    @BeforeAll
    fun setupDefaultOps() {
        println("All available platforms: \n" + PlatformProvider.getAvailablePlatforms().joinToString("\n"))
        println()

        val platformInfo = PlatformProvider.findPlatform { it.platformImplementationType == "OpenCL" && it.deviceType in listOf(DeviceType.CPU, DeviceType.GPU) }
        println("Running tests on platform:\n$platformInfo")

        platformKey = platformInfo.platformKey
        PlatformProvider.registerAsDefault(
            platformKey,
            PlatformProvider.tensorOperations(platformKey) as TensorOperations<AbstractRawTensor<Any>>,
            platformInfo
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
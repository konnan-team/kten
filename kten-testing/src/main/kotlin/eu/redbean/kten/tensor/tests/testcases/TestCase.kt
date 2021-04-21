package eu.redbean.kten.tensor.tests.testcases

import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.api.tensor.serialization.serializers.tensorFromJson
import javax.xml.bind.JAXBContext
import javax.xml.bind.annotation.*

@XmlRootElement(name = "test-case")
class TestCase(
    @field:XmlElement(name = "tensor") val tensors: MutableList<TestCaseTensorElement> = mutableListOf()
) {

    private lateinit var parsedTensors: Map<String, Tensor>

    private fun lateInit() {
        parsedTensors = tensors.map { it.name to Tensor.tensorFromJson(it.json) }.toMap()
        val gc = PlatformProvider.tensorOperations().garbageCollector()
        gc.mustKeep(*parsedTensors.values.toTypedArray())
    }

    operator fun get(name: String) = parsedTensors[name]

    companion object {

        private val context = JAXBContext.newInstance(TestCase::class.java)
        private val cache = mutableMapOf<String, TestCase>()

        fun loadTestCase(resourcePath: String): TestCase {
            return cache.computeIfAbsent(resourcePath) {
                TestCase::class.java.getResourceAsStream(it).use {
                    val testCase = context.createUnmarshaller().unmarshal(it) as TestCase
                    testCase.lateInit()
                    testCase
                }
            }
        }

    }

}

@XmlAccessorType(XmlAccessType.FIELD)
class TestCaseTensorElement(
    @XmlAttribute val name: String = "N/A",
    @XmlValue val json: String = "N/A"
)
package eu.redbean.kten.api.tensor.platform

import eu.redbean.kten.api.tensor.operations.BasicTensorOperations
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import org.reflections.Reflections
import java.util.*

object PlatformProvider {

    var epsilon = 1e-7f

    /**
     * If the platform implementation handles memory management directly, this value determines, approximately what "percentage" of the device memory can
     * be used. (Note: if this value is set for example to 0.2 it won't mean exactly 20% of memory will be used, because memory management depends on a
     * lot of factors. This value should be viewed as a hint to the platform implementation's garbage collector.)
     *
     * Value must be between 0.0 and 0.9
     *
     * @throws IllegalArgumentException if the value is not in the 0.0 to 0.9 range
     */
    var memoryUsageScaleHint = 0.85
        set(value) {
            if (value !in 0.0..0.9) {
                throw IllegalArgumentException("Memory usage scale value must be in range 0.0 to 0.9")
            }
            field = value
        }

    private val platforms = mutableMapOf<String, TensorOperations<AbstractRawTensor<Any>>>()

    private val platformInfos = mutableListOf<PlatformInfo>()

    private val platformTransformers = mutableMapOf<Pair<String, String>, (AbstractRawTensor<Any>) -> AbstractRawTensor<Any>>()

    var defaultPlatformKey = "NONE"
        private set

    init {
        println("Platform specific implementations found:")
        Reflections("eu.redbean.kten")
            .getSubTypesOf(PlatformInitializer::class.java)
            .forEach {
                if (it.kotlin.objectInstance != null)
                    println(it.kotlin.objectInstance!!.platformKeys)
            }
    }

    internal fun defaultOps(): TensorOperations<AbstractRawTensor<Any>> {
        return platforms[defaultPlatformKey] ?: throw IllegalStateException(
            "No default platform found. " +
                    "Make sure that backend.jvm module or other default platform module is added to classpath."
        )
    }

    internal fun platformOps(platform: String): TensorOperations<AbstractRawTensor<Any>> {
        return platforms[platform] ?: throw IllegalArgumentException("No platform with name: ${platform} found.")
    }

    fun tensorOperations(platform: String? = null): BasicTensorOperations {
        if (platform == null)
            return defaultOps()
        return platformOps(platform)
    }

    fun register(key: String, operations: TensorOperations<AbstractRawTensor<Any>>, platformInfo: PlatformInfo) {
        platforms[key] = operations
        platformInfos += platformInfo
    }

    fun registerAsDefault(key: String, operations: TensorOperations<AbstractRawTensor<Any>>, platformInfo: PlatformInfo) {
        register(key, operations, platformInfo)
        defaultPlatformKey = key
    }

    fun transformRawData(tensor: AbstractRawTensor<Any>, fromPlatform: String, toPlatform: String): AbstractRawTensor<Any> {
        val key = fromPlatform to toPlatform
        if (platformTransformers.containsKey(key)) {
            return platformTransformers[key]!!.invoke(tensor)
        }
        throw IllegalStateException("Cannot transform data from ${fromPlatform} to ${toPlatform}, because there is no appropriate transformer registered.")
    }

    fun registerPlatformTransformer(fromAndToPlatform: Pair<String, String>, transformer: (AbstractRawTensor<Any>) -> AbstractRawTensor<Any>) {
        platformTransformers.put(fromAndToPlatform, transformer)
    }

    fun findPlatform(platformSelector: (PlatformInfo) -> Boolean): PlatformInfo {
        return platformInfos.find(platformSelector) ?: throw IllegalArgumentException("No platform found for the specified selector")
    }

    fun findAllPlatforms(platformSelector: (PlatformInfo) -> Boolean): List<PlatformInfo> {
        return platformInfos.filter(platformSelector)
    }

    fun getAvailablePlatforms(): List<PlatformInfo> {
        return Collections.unmodifiableList(platformInfos)
    }

}
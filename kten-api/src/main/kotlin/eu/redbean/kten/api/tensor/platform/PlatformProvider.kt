package eu.redbean.kten.api.tensor.platform

import eu.redbean.kten.api.tensor.operations.BasicTensorOperations
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import org.reflections.Reflections

object PlatformProvider {

    var epsilon = 1e-7f

    private val platforms = mutableMapOf<String, TensorOperations<AbstractRawTensor<Any>>>()

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
        return platforms[defaultPlatformKey] ?: throw IllegalStateException("No default platform found. " +
                "Make sure that backend.jvm module or other default platform module is added to classpath.")
    }

    internal fun platformOps(platform: String): TensorOperations<AbstractRawTensor<Any>> {
        return platforms[platform] ?: throw IllegalArgumentException("No platform with name: ${platform} found.")
    }

    fun tensorOperations(platform: String? = null): BasicTensorOperations {
        if (platform == null)
            return defaultOps()
        return platformOps(platform)
    }

    fun register(key: String, operations: TensorOperations<AbstractRawTensor<Any>>) {
        platforms[key] = operations
    }

    fun registerAsDefault(key: String, operations: TensorOperations<AbstractRawTensor<Any>>) {
        register(key, operations)
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

}
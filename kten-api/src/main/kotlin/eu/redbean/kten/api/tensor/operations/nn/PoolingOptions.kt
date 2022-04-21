package eu.redbean.kten.api.tensor.operations.nn

enum class PoolingType {
    MAX, AVG
}

data class PoolingOptions(
    val type: PoolingType,
    val useCeil: Boolean = false,
    val includePadding: Boolean = false
)
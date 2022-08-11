package eu.redbean.kten.api.tensor.operations

interface PlatformMetricsProvider {

    fun getMetrics(): Map<String, Float>

}
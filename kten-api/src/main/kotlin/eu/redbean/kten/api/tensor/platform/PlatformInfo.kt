package eu.redbean.kten.api.tensor.platform

data class PlatformInfo(
    val platformImplementationType: String,
    val deviceType: DeviceType,
    val availableMemory: Long,
    val platformKey: String,
    val platformSpecificInfo: PlatformSpecificDeviceInfo? = null
) {

    private val memoryRegex = "([0-9]+)\\s*(GB|GIGS|MB|MEGS|KB|KILOS)".toRegex()

    override fun toString(): String {
        return """Platform implementation: $platformImplementationType
             |Device type: $deviceType
             |Available memory: ${availableMemory / 1024 / 1024} MB
             |Platform key: $platformKey
             |Platform specific info: $platformSpecificInfo
        """.trimMargin()
    }

    private fun unitScale(unit: String): Long = when (unit) {
        "GB", "GIGS" -> 1_073_741_824L
        "MB", "MEGS" -> 1_048_576L
        "KB", "KILOS" -> 1024L
        else -> 0L
    }

    private fun matchMemory(memory: String, matcher: (Long, Long) -> Boolean): Boolean {
        val match = memoryRegex.matchEntire(memory.toUpperCase().trim())
        if (match != null) {
            val (value, unit) = match.destructured
            return matcher(availableMemory, value.toLong() * unitScale(unit))
        }
        return false
    }

    infix fun hasMoreMemoryThan(memory: String) = matchMemory(memory) { a, b -> a > b }

    infix fun hasLessMemoryThan(memory: String) = matchMemory(memory) { a, b -> a < b }

}
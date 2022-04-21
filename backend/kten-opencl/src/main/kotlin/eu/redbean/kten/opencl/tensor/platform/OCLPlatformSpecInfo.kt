package eu.redbean.kten.opencl.tensor.platform

import eu.redbean.kten.api.tensor.platform.PlatformSpecificDeviceInfo
import org.jocl.cl_device_id

class OCLPlatformSpecInfo(
    val maxWorkGroupSize: Long,
    val maxWorkItemSizes: List<Long>,
    val deviceId: cl_device_id
): PlatformSpecificDeviceInfo {

    override fun toString(): String {
        return "Max work group size: $maxWorkGroupSize max work item sizes: $maxWorkItemSizes"
    }
}
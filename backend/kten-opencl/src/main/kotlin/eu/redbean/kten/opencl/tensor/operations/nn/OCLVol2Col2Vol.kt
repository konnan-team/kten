package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject
import eu.redbean.kten.opencl.tensor.store.OCLStoreView

class OCLVol2Col2Vol(
    val kernelDepth: Int, val kernelHeight: Int, val kernelWidth: Int,
    val paddingDepth: Int, val paddingHeight: Int, val paddingWidth: Int,
    val strideDepth: Int, val strideHeight: Int, val strideWidth: Int,
    val dilationDepth: Int, val dilationHeight: Int, val dilationWidth: Int,
    val ops: OCLTensorOperations
) {

    fun vol2col(
        vol: OCLStoreView,
        channels: Int, depth: Int, height: Int, width: Int,
        col: OCLMemoryObject
    ) {
        ops.environment.kernelStore.vol2col(
            vol,
            kernelDepth, kernelHeight, kernelWidth,
            paddingDepth, paddingHeight, paddingWidth,
            strideDepth, strideHeight, strideWidth,
            dilationDepth, dilationHeight, dilationWidth,
            channels,
            depth, height, width,
            col
        )
    }

    fun col2vol(
        col: OCLMemoryObject,
        channels: Int, depth: Int, height: Int, width: Int,
        outputDepth: Int, outputHeight: Int, outputWidth: Int,
        vol: OCLStoreView
    ) {
        ops.environment.kernelStore.col2vol(
            col,
            kernelDepth, kernelHeight, kernelWidth,
            paddingDepth, paddingHeight, paddingWidth,
            strideDepth, strideHeight, strideWidth,
            dilationDepth, dilationHeight, dilationWidth,
            channels,
            depth, height, width,
            outputDepth, outputHeight, outputWidth,
            vol
        )
    }

}
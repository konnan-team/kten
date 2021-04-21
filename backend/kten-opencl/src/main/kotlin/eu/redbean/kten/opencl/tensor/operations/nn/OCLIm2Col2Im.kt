package eu.redbean.kten.opencl.tensor.operations.nn

import eu.redbean.kten.opencl.tensor.operations.OCLTensorOperations
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.SOURCE
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.TARGET
import eu.redbean.kten.opencl.tensor.store.OCLStoreView
import org.jocl.blast.CLBlast
import org.jocl.blast.CLBlastKernelMode

class OCLIm2Col2Im(
    val kernelHeight: Int, val kernelWidth: Int,
    val paddingHeight: Int, val paddingWidth: Int,
    val strideHeight: Int, val strideWidth: Int,
    val dilationHeight: Int, val dilationWidth: Int,
    val ops: OCLTensorOperations
) {

    fun im2col(im: OCLStoreView,
               channels: Int, height: Int, width: Int,
               col: OCLMemoryObject) {
        CLBlast.CLBlastSim2col(
            CLBlastKernelMode.CLBlastKernelModeCrossCorrelation,
            channels.toLong(), height.toLong(), width.toLong(),
            kernelHeight.toLong(), kernelWidth.toLong(),
            paddingHeight.toLong(), paddingWidth.toLong(),
            strideHeight.toLong(), strideWidth.toLong(),
            dilationHeight.toLong(), dilationWidth.toLong(),
            im.getMemoryObject(SOURCE), im.offset,
            col.getMemoryObject(TARGET), 0L,
            ops.environment.commandQueue,
            null
        )
    }

    fun col2im(col: OCLMemoryObject,
               channels: Int, height: Int, width: Int,
               im: OCLStoreView) {
        im.fill(0f)

        CLBlast.CLBlastScol2im(
            CLBlastKernelMode.CLBlastKernelModeCrossCorrelation,
            channels.toLong(), height.toLong(), width.toLong(),
            kernelHeight.toLong(), kernelWidth.toLong(),
            paddingHeight.toLong(), paddingWidth.toLong(),
            strideHeight.toLong(), strideWidth.toLong(),
            dilationHeight.toLong(), dilationWidth.toLong(),
            col.getMemoryObject(SOURCE), 0L,
            im.getMemoryObject(TARGET), im.offset,
            ops.environment.commandQueue,
            null
        )
    }

    fun col2im(col: OCLMemoryObject,
               channels: Int, height: Int, width: Int,
               outputHeight: Int, outputWidth: Int,
               im: OCLStoreView) {
        im.fill(0f)

        ops.environment.kernelStore.col2imForTranspose(
            col,
            kernelHeight, kernelWidth,
            paddingHeight, paddingWidth,
            strideHeight, strideWidth,
            dilationHeight, dilationWidth,
            channels, height, width,
            outputHeight, outputWidth,
            im
        )
    }

}
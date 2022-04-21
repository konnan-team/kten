package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.StoreView
import java.lang.Integer.min
import java.util.stream.IntStream

class JVMSpatialMaxPooling(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    dilationHeight: Int, dilationWidth: Int,
    useCeil: Boolean,
    ops: AbstractJVMTensorOperations
) : JVMSpatialPoolingBase(
    kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth, useCeil, true, ops
) {

    override fun updateOutputSliceView(input: StoreView, output: StoreView, indices: StoreView?) {
        IntStream.range(0, inputPlane).parallel().forEach {
            val inputStartPos = it * inputHeight * inputWidth
            for (i in 0 until outputHeight) {
                for (j in 0 until outputWidth) {
                    var hStart = i * strideHeight - paddingHeight
                    var wStart = j * strideWidth - paddingWidth
                    val hEnd = min(hStart + (kernelHeight - 1) * dilationHeight + 1, inputHeight)
                    val wEnd = min(wStart + (kernelWidth - 1) * dilationWidth + 1, inputWidth)
                    while (hStart < 0) hStart += dilationHeight
                    while (wStart < 0) wStart += dilationWidth

                    val outputStartPos = it * outputHeight * outputWidth + i * outputWidth + j

                    var maxIndex = -1f
                    var maxVal = Float.NEGATIVE_INFINITY

                    for (y in hStart until hEnd step dilationHeight) {
                        for (x in wStart until wEnd step dilationWidth) {
                            val idx = y * inputWidth + x
                            val value = input[inputStartPos + idx]
                            if (value > maxVal) {
                                maxVal = value
                                maxIndex = idx.toFloat()
                            }
                        }
                    }

                    output[outputStartPos] = maxVal
                    indices!![outputStartPos] = maxIndex
                }
            }
        }
    }

    override fun updateGradSliceView(gradIn: StoreView, gradOut: StoreView, indices: StoreView?) {
        IntStream.range(0, inputPlane).parallel().forEach {
            val gradInStartPos = it * inputHeight * inputWidth
            val gradOutStartPos = it * outputHeight * outputWidth

            for (i in 0 until outputHeight) {
                for (j in 0 until outputWidth) {
                    val localPos = i * outputWidth + j
                    val gradIndex = indices!![gradOutStartPos + localPos].toInt()
                    if (gradIndex != -1) {
                        gradIn[gradInStartPos + gradIndex] += gradOut[gradOutStartPos + localPos]
                    }
                }
            }
        }
    }

}
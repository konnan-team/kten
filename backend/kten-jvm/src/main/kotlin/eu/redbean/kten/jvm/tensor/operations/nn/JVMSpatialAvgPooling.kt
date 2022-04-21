package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.StoreView
import java.util.stream.IntStream
import kotlin.math.max
import kotlin.math.min

class JVMSpatialAvgPooling(
    kernelHeight: Int, kernelWidth: Int,
    paddingHeight: Int, paddingWidth: Int,
    strideHeight: Int, strideWidth: Int,
    useCeil: Boolean,
    val includePadding: Boolean,
    ops: AbstractJVMTensorOperations
) : JVMSpatialPoolingBase(
    kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth, 1, 1, useCeil, false, ops
) {

    override fun updateOutputSliceView(input: StoreView, output: StoreView, indices: StoreView?) {
        IntStream.range(0, inputPlane).parallel().forEach {
            val inputStartPos = it * inputHeight * inputWidth
            val outputStartPos = it * outputHeight * outputWidth
            for (i in 0 until outputHeight) {
                for (j in 0 until outputWidth) {
                    var hStart = i * strideHeight - paddingHeight
                    var wStart = j * strideWidth - paddingWidth
                    var hEnd = Integer.min(hStart + kernelHeight, inputHeight + paddingHeight)
                    var wEnd = Integer.min(wStart + kernelWidth , inputWidth + paddingWidth)

                    val poolSize = (hEnd - hStart) * (wEnd - wStart)

                    hStart = max(hStart, 0)
                    wStart = max(wStart, 0)
                    hEnd = min(hEnd, inputHeight)
                    wEnd = min(wEnd, inputWidth)

                    val itemsInPool = if (includePadding) poolSize else (hEnd - hStart) * (wEnd - wStart)

                    var sum = 0f

                    for (y in hStart until hEnd) {
                        for (x in wStart until wEnd) {
                            sum += input[inputStartPos + y * inputWidth + x]
                        }
                    }

                    val outputPos = outputStartPos + i * outputWidth + j

                    output[outputPos] = sum / itemsInPool
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
                    var hStart = i * strideHeight - paddingHeight
                    var wStart = j * strideWidth - paddingWidth
                    var hEnd = Integer.min(hStart + kernelHeight, inputHeight + paddingHeight)
                    var wEnd = Integer.min(wStart + kernelWidth , inputWidth + paddingWidth)

                    val poolSize = (hEnd - hStart) * (wEnd - wStart)

                    hStart = max(hStart, 0)
                    wStart = max(wStart, 0)
                    hEnd = min(hEnd, inputHeight)
                    wEnd = min(wEnd, inputWidth)

                    val itemsInPool = if (includePadding) poolSize else (hEnd - hStart) * (wEnd - wStart)

                    val gradOutPos = gradOutStartPos + i * outputWidth + j

                    for (y in hStart until hEnd) {
                        for (x in wStart until wEnd) {
                            gradIn[gradInStartPos + y * inputWidth + x] += gradOut[gradOutPos] / itemsInPool
                        }
                    }
                }
            }
        }
    }


}
package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.jvm.tensor.store.StoreView
import java.util.stream.IntStream

class Im2Col2Im(
    val kernelHeight: Int, val kernelWidth: Int,
    val paddingHeight: Int, val paddingWidth: Int,
    val strideHeight: Int, val strideWidth: Int,
    val dilationHeight: Int, val dilationWidth: Int
) {

    private fun im2col2imIndex(
        channels: Int, colHeight: Int, colWidth: Int,
        op: (colC: Int, colH: Int, colW: Int, imC: Int, imH: Int, imW: Int) -> Unit
    ) {
        val colChannels = channels * kernelHeight * kernelWidth

        IntStream.range(0, colChannels)
            .forEach { colC ->
                val offsetW = colC % kernelWidth
                val offsetH = (colC / kernelWidth) % kernelHeight
                val imC = colC / kernelHeight / kernelWidth

                IntStream.range(0, colHeight).parallel()
                    .forEach { colH ->
                        IntStream.range(0, colWidth).parallel()
                            .forEach { colW ->
                                val imH = colH * strideHeight - paddingHeight + offsetH * dilationHeight
                                val imW = colW * strideWidth - paddingWidth + offsetW * dilationWidth
                                op(colC, colH, colW, imC, imH, imW)
                            }
                    }
            }
    }

    fun im2col(im: StoreView,
               channels: Int, height: Int, width: Int,
               col: FloatArray) {
        val colHeight = (height + 2 * paddingHeight - (dilationHeight * (kernelHeight - 1) + 1)) / strideHeight + 1
        val colWidth = (width + 2 * paddingWidth - (dilationWidth * (kernelWidth - 1) + 1)) / strideWidth + 1

        im2col2imIndex(channels, colHeight, colWidth) { colC, colH, colW, imC, imH, imW ->
            col[(colC * colHeight + colH) * colWidth + colW] =
                if ((imH in 0 until height) && (imW in 0 until width))
                    im[(imC * height + imH) * width + imW]
                else
                    0f
        }
    }

    fun col2im(col: FloatArray,
               channels: Int, height: Int, width: Int,
               outputHeight: Int, outputWidth: Int,
               im: StoreView
    ) {
        im.fill(0f)

        im2col2imIndex(channels, outputHeight, outputWidth) { colC, colH, colW, imC, imH, imW ->
            if ((imH in 0 until height) && (imW in 0 until width)) {
                im[(imC * height + imH) * width + imW] += col[(colC * outputHeight + colH) * outputWidth + colW]
            }
        }
    }

}
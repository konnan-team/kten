package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.jvm.tensor.store.StoreView
import java.util.stream.IntStream

class Vol2Col2Vol(
    val kernelDepth: Int, val kernelHeight: Int, val kernelWidth: Int,
    val paddingDepth: Int, val paddingHeight: Int, val paddingWidth: Int,
    val strideDepth: Int, val strideHeight: Int, val strideWidth: Int,
    val dilationDepth: Int, val dilationHeight: Int, val dilationWidth: Int,
) {

    private fun vol2col2volIndex(
        channels: Int, colDepth: Int, colHeight: Int, colWidth: Int,
        op: (colC: Int, colD: Int, colH: Int, colW: Int, volC: Int, volD: Int, volH: Int, volW: Int) -> Unit
    ) {
        val colChannels = channels * kernelDepth * kernelHeight * kernelWidth

        IntStream.range(0, colChannels).parallel()
            .forEach { colC ->
                val offsetW = colC % kernelWidth
                val offsetH = (colC / kernelWidth) % kernelHeight
                val offsetD = (colC / kernelWidth / kernelHeight) % kernelDepth
                val volC = colC / kernelDepth / kernelHeight / kernelWidth

                IntStream.range(0, colDepth).parallel()
                    .forEach { colD ->
                        IntStream.range(0, colHeight).parallel()
                            .forEach { colH ->
                                IntStream.range(0, colWidth).parallel()
                                    .forEach { colW ->
                                        val volD = colD * strideDepth - paddingDepth + offsetD + dilationDepth
                                        val volH = colH * strideHeight - paddingHeight + offsetH * dilationHeight
                                        val volW = colW * strideWidth - paddingWidth + offsetW * dilationWidth
                                        op(colC, colD, colH, colW, volC, volD, volH, volW)
                                    }
                            }
                    }
            }
    }

    fun vol2col(vol: StoreView,
               channels: Int, depth: Int, height: Int, width: Int,
               col: FloatArray) {
        val colDepth = (depth + 2 * paddingDepth - (dilationDepth * (kernelDepth -1) + 1)) / strideDepth + 1
        val colHeight = (height + 2 * paddingHeight - (dilationHeight * (kernelHeight - 1) + 1)) / strideHeight + 1
        val colWidth = (width + 2 * paddingWidth - (dilationWidth * (kernelWidth - 1) + 1)) / strideWidth + 1

        vol2col2volIndex(channels, colDepth , colHeight, colWidth) { colC, colD, colH, colW, volC, volD, volH, volW ->
            col[((colC * colDepth + colD) * colHeight + colH) * colWidth + colW] =
                if ((volD in 0 until depth) && (volH in 0 until height) && (volW in 0 until width))
                    vol[((volC * depth + volD) * height + volH) * width + volW]
                else
                    0f
        }
    }

    fun col2vol(col: FloatArray,
               channels: Int, depth: Int, height: Int, width: Int,
               outputDepth: Int, outputHeight: Int, outputWidth: Int,
               vol: StoreView
    ) {
        vol.fill(0f)

        vol2col2volIndex(channels, outputDepth, outputHeight, outputWidth) { colC, colD, colH, colW, volC, volD, volH, volW ->
            if ((volD in 0 until depth) && (volH in 0 until height) && (volW in 0 until width)) {
                vol[((volC * depth + volD) * height + volH) * width + volW] += col[((colC * outputDepth + colD) * outputHeight + colH) * outputWidth + colW]
            }
        }
    }

}
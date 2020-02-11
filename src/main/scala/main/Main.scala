package main

//LD_LIBRARY_PATH=/opt/opencv/share/OpenCV/opencv_3.4/lib
//-Djava.library.path=/opt/opencv/share/OpenCV/opencv_3.4/share/OpenCV/java
import org.opencv.core.{Core, CvType, Mat, MatOfByte}
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.videoio.{VideoCapture, Videoio}
import org.tensorflow.SavedModelBundle
import org.tensorflow.Tensor
import org.tensorflow.types.UInt8
import com.google.protobuf.TextFormat
import object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMap
import object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMapItem
import org.tensorflow.SavedModelBundle
import org.tensorflow.Tensor
import org.tensorflow.framework.MetaGraphDef
import org.tensorflow.framework.SignatureDef
import org.tensorflow.framework.TensorInfo
import org.tensorflow.types.UInt8
import javax.imageio.ImageIO
import java.awt.Graphics2D
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.{ByteArrayInputStream, File, IOException, PrintStream}
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths
import java.util

import scala.io.Source
import scala.util.control.Breaks

object Main {
  @throws[Exception]
  def main(args: Array[String]): Unit = {

    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val cap = new VideoCapture("rtsp://172.16.165.133:8554/")
    HighGui.namedWindow("Video Feed", HighGui.WINDOW_AUTOSIZE)
    var frame = new Mat()
//    while(true){
//      cap.read(frame)
//      HighGui.imshow("Video Feed", frame)
//      HighGui.waitKey(10)
//    }

//    for (i <- 1 to 10){
//      cap.read(frame)
//      Imgcodecs.imwrite(s"/mnt/ramdisk/image$i.png", frame)
//    }
    //	cap.release()

    val labels = loadLabels(getClass.getResource("/labels/mscoco_label_map.pbtxt").getPath)
    try {
//      val model = SavedModelBundle.load(getClass.getResource("/ssd_inception_v2_coco_2017_11_17/saved_model").getPath, "serve")
      val model = SavedModelBundle.load(getClass.getResource("/ssdte  _mobilenet_v1_coco_2018_01_28/saved_model").getPath, "serve")
      try {
        //        printSignature(model)
        for (i <- 1 to 30) {
          val before = System.currentTimeMillis()
          cap.read(frame)
          var mob = new MatOfByte()
          Imgcodecs.imencode(".jpg", frame, mob)
          var outputs: util.List[Tensor[_]] = null
          try {
            val input = makeImageTensor(mob)
            try outputs = model.session.runner.feed("image_tensor", input).fetch("detection_scores").fetch("detection_classes").fetch("detection_boxes").run
            catch { case e: Exception => System.err.println(e)}
          } catch { case e: Exception => System.err.println(e)}
          try {
            val scoresT = outputs.get(0)
            val classesT = outputs.get(1)
            val boxesT = outputs.get(2)
            try { // All these tensors have:
              // - 1 as the first dimension
              // - maxObjects as the second dimension
              // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
              // This can be verified by looking at scoresT.shape() etc.
              val maxObjects = scoresT.shape()(1).toInt
              val scores: Array[Float] = scoresT.copyTo(Array.ofDim[Float](1, maxObjects))(0)
              val classes: Array[Float] = classesT.copyTo(Array.ofDim[Float](1, maxObjects))(0)
              val boxes: Array[Array[Float]] = boxesT.copyTo(Array.ofDim[Float](1, maxObjects, 4))(0)
              // Print all objects whose score is at least 0.5.
//              printf("* %s\n", filename)
              var foundSomething = false
              for (i <- 0 until scores.length) {
                Breaks.breakable{
                  if (scores(i) < 0.5) Breaks.break
                  foundSomething = true
                  printf("\tFound %-20s (score: %.4f)\n", labels(classes(i).toInt), scores(i))
                }
              }
              if (!foundSomething) println("No objects detected with a high enough score.")
            } catch { case e: Exception => System.err.println(e)}
          } catch { case e: Exception => System.err.println(e)}
          println(s"time: ${System.currentTimeMillis() - before}")
        }
      } catch { case e: Exception => System.err.println(e)}
    } catch { case e: Exception => System.err.println(e)}

  }

  @throws[Exception]
  def printSignature(model: SavedModelBundle): Unit = {
    val m = MetaGraphDef.parseFrom(model.metaGraphDef)
    val sig = m.getSignatureDefOrThrow("serving_default")
    val numInputs = sig.getInputsCount
    var i = 1
    println("MODEL SIGNATURE")
    println("Inputs:")

    sig.getInputsMap.entrySet.forEach(entry => {
      val t = entry.getValue
      printf("%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n", {
        i += 1; i - 1
      }, numInputs, entry.getKey, t.getName, t.getDtype)
    })
    val numOutputs = sig.getOutputsCount
    i = 1
    println("Outputs:")
    sig.getOutputsMap.entrySet.forEach(entry => {
      val t = entry.getValue
      printf("%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n", {
        i += 1; i - 1
      }, numOutputs, entry.getKey, t.getName, t.getDtype)
    })
    println("-----------------------------------------------")
  }

  @throws[Exception]
  def loadLabels(filename: String) = {
    val text = new String(Files.readAllBytes(Paths.get(filename)), StandardCharsets.UTF_8)
    val builder = StringIntLabelMap.newBuilder
    TextFormat.merge(text, builder)
    val proto = builder.build
    var maxId = 0
    proto.getItemList.forEach(item => {
      if (item.getId > maxId) maxId = item.getId
    })
    val ret = new Array[String](maxId + 1)
    proto.getItemList.forEach(item => {
      ret(item.getId) = item.getDisplayName
    })
    ret
  }

  private def bgr2rgb(data: Array[Byte]): Unit = {
    var i = 0
    while ( {
      i < data.length
    }) {
      val tmp = data(i)
      data(i) = data(i + 2)
      data(i + 2) = tmp

      i += 3
    }
  }

  @throws[IOException]
  def makeImageTensor(mob: MatOfByte) = {
    var img = ImageIO.read(new ByteArrayInputStream(mob.toArray))
    if (img.getType != BufferedImage.TYPE_3BYTE_BGR) {
      val newImage = new BufferedImage(img.getWidth, img.getHeight, BufferedImage.TYPE_3BYTE_BGR)
      val g = newImage.createGraphics
      g.drawImage(img, 0, 0, img.getWidth, img.getHeight, null)
      g.dispose()
      img = newImage
    }
    val data = img.getData.getDataBuffer.asInstanceOf[DataBufferByte].getData
    // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
    bgr2rgb(data)
    val BATCH_SIZE = 1
    val CHANNELS = 3
    val shape = Array[Long](BATCH_SIZE, img.getHeight, img.getWidth, CHANNELS)
    Tensor.create(classOf[UInt8], shape, ByteBuffer.wrap(data))
  }

  def printUsage(s: PrintStream): Unit = {
    s.println("USAGE: <model> <label_map> <image> [<image>] [<image>]")
    s.println("")
    s.println("Where")
    s.println("<model> is the path to the SavedModel directory of the model to use.")
    s.println("        For example, the saved_model directory in tarballs from ")
    s.println("        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)")
    s.println("")
    s.println("<label_map> is the path to a file containing information about the labels detected by the model.")
    s.println("            For example, one of the .pbtxt files from ")
    s.println("            https://github.com/tensorflow/models/tree/master/research/object_detection/data")
    s.println("")
    s.println("<image> is the path to an image file.")
    s.println("        Sample images can be found from the COCO, Kitti, or Open Images dataset.")
    s.println("        See: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md")
  }


}


// camera page
import 'dart:async';
import 'dart:convert'; //to convert base64 -- json
import 'dart:io'; //to process files and websocket of dart
import 'dart:typed_data';//to process bytes  (voice)
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:web_socket_channel/io.dart'; //webSocket channel
import 'package:http/http.dart' as http; //to send voice request
import 'package:audioplayers/audioplayers.dart'; //to play sound
import 'package:path_provider/path_provider.dart';//temporary folder for voice files

List<CameraDescription> cameras = [];//list for available cameras

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? controller; //to control the camera
  IOWebSocketChannel? channel; //to control WebSocket communication
  Timer? captureTimer;
  bool isSending = false;
  bool streaming = false;

  String serverIp = "192.168.100.18";
  int serverPort = 8000;

  String currentChar = "";
  String sentence = "";
  String status = "غير متصل";

  int currentCameraIndex = 0;

  @override
  void initState() { //initialize camera and connection when the screen opens
    super.initState();
    _initAll();
  }

  Future<void> _initAll() async { //async function to initialize camera
    try {
      cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        final frontCamera = cameras.firstWhere(//firstly looking for front camera
              (cam) => cam.lensDirection == CameraLensDirection.front,
          orElse: () => cameras.first,
        );
        controller = CameraController(frontCamera, ResolutionPreset.medium, enableAudio: false);
        await controller!.initialize();
        setState(() {});
      }
    } catch (e) {
      debugPrint("Camera init error: $e");
    }

    _connectWebSocket();
  }

  Future<void> _connectWebSocket() async {
    try {
      final uri = Uri.parse("ws://$serverIp:$serverPort/ws"); //connection URL
      final socket = await WebSocket.connect(uri.toString());//connect to websocket
      channel = IOWebSocketChannel(socket);//R&W channel
      setState(() => status = "متصل");

      channel!.stream.listen((message) { //listen for messages
        try {
          final data = jsonDecode(message);//convert json to map
          if (data.containsKey("result")) {
            setState(() {
              currentChar = data["result"] ?? "";
            });
          } else if (data.containsKey("error")) {
            setState(() => status = "Server error: ${data['error']}");
          }
        } catch (e) {
          debugPrint("Invalid message: $e");
        }
      }, onDone: () { //if server closed

        debugPrint("Server closed connection");
        channel = null;                   // مهم جداً
        stopPeriodicCapture();            // يوقف البث
        setState(() {
          streaming = false;
          status = "غير متصل";
        });
      },onError: (err) {
        debugPrint("WebSocket error: $err");
        channel = null;                   // مهم جداً
        stopPeriodicCapture();
        setState(() {
          streaming = false;
          status = "Connection error";
        });
      },
      );


      startPeriodicCapture();
      setState(() => streaming = false);
    } catch (e) {
      debugPrint("WebSocket connect failed: $e");
      setState(() => status = "فشل الإتصال");
    }
  }


  void startPeriodicCapture() {
    captureTimer?.cancel();
    captureTimer = Timer.periodic(const Duration(milliseconds: 200), (_) async {
      if (!streaming || // streaming stoped
          isSending || //another frame is sending
          controller == null || //camera not initialized
          !controller!.value.isInitialized)
      {return;}
      if (channel == null) return;//no websocket connection

      isSending = true;
      try {
        final XFile file = await controller!.takePicture(); //take picture
        final bytes = await file.readAsBytes(); //convert image to bytes
        final base64Str = base64Encode(bytes);//convert it to base64
        final payload = jsonEncode({"frame": base64Str});//يجهزها في json to send it to server
        channel!.sink.add(payload); //send it to server
        final f = File(file.path);
        if (await f.exists()) await f.delete(); //delete the image
      } catch (e) {
        debugPrint("Capture/send error: $e");
      } finally {
        isSending = false;
      }
    });
  }


  void stopPeriodicCapture() {
    captureTimer?.cancel();
  }

  void onAdd() { //to add the letter
    if (currentChar.isNotEmpty) {
      setState(() {
        sentence += currentChar;
      });
    }
  }

  void onRemove() { //to remove the letter
    if (sentence.isNotEmpty) {
      setState(() {
        sentence = sentence.substring(0, sentence.length - 1);
      });
    }
  }

  void onClear() { //to clear the sentence
    setState(() => sentence = "");
  }

  // دالة لتشغيل الصوت من بايتات
  Future<void> playAudioFromBytes(Uint8List bytes) async {
    final dir = await getTemporaryDirectory(); //temporary file in mobile
    final file = File('${dir.path}/tts_audio.mp3');//ينشئ ملف MP3
    await file.writeAsBytes(bytes, flush: true);//writes bytes to file

    final player = AudioPlayer();
    await player.play(DeviceFileSource(file.path));//play voice file
  }

  Future<void> onSpeak() async {
    if (sentence.trim().isEmpty) return;
    try {
      final url = Uri.parse("http://$serverIp:$serverPort/tts");//HTTP URL for TTS
      final res = await http.post( //send json contains the text
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"text": sentence}),
      );

      final data = jsonDecode(res.body);//read response from server
      if (data["audio"] != null) {
        final bytes = base64Decode(data["audio"]);//convert it to Uint8List
        await playAudioFromBytes(bytes);
      } else {
        debugPrint("TTS error: ${data['error']}");//if error appear
      }
    } catch (e) {
      debugPrint("Speak error: $e");//any error
    }
  }

  bool get isConnected => channel != null;

  void onStop() async {
    if (!isConnected) {  // if no connection
      setState(() {
        streaming = false;
        status = "غير متصل بالسيرفر";
      });
      debugPrint("Cannot start streaming: not connected to server");
      return;
    }

    if (!streaming) { //streaming not started
      startPeriodicCapture();
      setState(() {
        streaming = true;
        status = "جاري الإرسال";
      });
    } else {
      stopPeriodicCapture(); //stop the streaming
      setState(() {
        streaming = false;
        status = "متوقف مؤقتا";
      });
    }
  }


  Future<void> switchCamera() async { //to switch the current camera
    if (cameras.isEmpty) return;

    // select the next camera
    currentCameraIndex = (currentCameraIndex + 1) % cameras.length;
    final newCamera = cameras[currentCameraIndex];

    // switch the camera without stop the streaming
    final wasStreaming = streaming;
    streaming = false; // مؤقتًا لإيقاف الإرسال أثناء التبديل
    await controller?.dispose();
    controller = CameraController(newCamera, ResolutionPreset.medium, enableAudio: false);
    await controller!.initialize();
    streaming = wasStreaming;

    setState(() {});//rebuild UI
  }

  void _showInfoDialog(BuildContext context) { //show information dialog of buttons functionalities
    showDialog( //popup dialog
      context: context,
      builder: (BuildContext context) {
        return AlertDialog( //dialog with rounded corners
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
          title: const Text("معلومات الأزرار", textAlign: TextAlign.center, style: TextStyle(fontWeight: FontWeight.bold,
            fontFamily: "El_Messiri",)
          ),

          content: const Column(
            mainAxisSize: MainAxisSize.min, //to make the dialog size match the content
            crossAxisAlignment: CrossAxisAlignment.start, //content start from the start (Right in RTL layout)
            children: [
              Directionality(
                textDirection: TextDirection.rtl, // align all content from left to right
                child: Column(
                  mainAxisSize: MainAxisSize.min,//column height suits the content
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(height: 8),
                    Row(
                      children: [
                        Icon(Icons.add, size: 22, color: Colors.green,),
                        SizedBox(width: 8),
                        Text(" الإضافة: لإضافة حرف جديد.",style: TextStyle(
                            fontFamily: 'El_Messiri',
                            fontSize: 14),),],),
                    SizedBox(height: 8),

                    Row(children: [
                      Icon(Icons.remove, size: 22, color: Colors.deepOrangeAccent,),
                      SizedBox(width: 8),
                      Text(" الطرح: لإزالة آخر حرف.",style: TextStyle(
                          fontFamily: 'El_Messiri',
                          fontSize: 14),)
                    ],)
                    ,
                    SizedBox(height: 8),

                    Row(children: [
                      Icon(Icons.delete_outline,size: 22, color: Colors.red,),
                      SizedBox(width: 8),
                      Text(" الحذف: لحذف جميع الحروف.",style: TextStyle(
                          fontFamily: 'El_Messiri',
                          fontSize: 14),)],)
                    ,
                    SizedBox(height: 8),

                    Row(children: [
                      Icon(Icons.volume_up,size: 22,color: Colors.blue,),
                      SizedBox(width: 8)
                      ,Text(" الصوت: لتشغيل نطق الجملة.",style: TextStyle(
                          fontFamily: 'El_Messiri',
                          fontSize: 14),)],)
                    ,
                    SizedBox(height: 8),

                    Row(children: [Icon(Icons.stop_circle, size: 22,),
                      SizedBox(height: 8),
                      Expanded(child: Text("  الإيقاف: لإيقاف/تشغيل البث.",style: TextStyle(
                          fontFamily: 'El_Messiri',
                          fontSize: 14),),)
                    ],),

                    SizedBox(height: 8),
                    Row(
                      children: [
                        Icon(Icons.cameraswitch, size: 22, color: Colors.grey,),
                        SizedBox(width: 8),
                        Expanded(
                          child: Text(" تبديل الكاميرا: للتبديل بين الكاميرا الأمامية والخلفية.",style: TextStyle(
                              fontFamily: 'El_Messiri',
                              fontSize: 14),),)
                      ],
                    )
                    ,
                  ],
                ),
              ),

            ],
          ),
          actionsAlignment: MainAxisAlignment.center,
          actions: [ //close button
            TextButton(onPressed: () => Navigator.pop(context), child: const Text("إغلاق",style: TextStyle(
              fontFamily: 'El_Messiri',
              fontSize: 18,
              color: Colors.black,
            ),),),
          ],
        );
      },
    );
  }

  @override
  void dispose() { //closes all active resources when leaving the page
    captureTimer?.cancel();
    controller?.dispose();
    channel?.sink.close();
    super.dispose();
  }

  Widget cameraButton(IconData icon, VoidCallback onTap, Color iconColor) { //reusable styling camera button
    return GestureDetector( //clickable button
      onTap: onTap,
      child: Container( height: 50, width: 49,
        decoration: BoxDecoration(
          color: Colors.black45,            // خلفية نصف شفافة
          borderRadius: BorderRadius.circular(14),
        ),
        child: Icon( icon,
          color: iconColor,
          size: 30,
        ),
      ),
    );
  }

  Color getStatusColor() { //return state indicator color
    if (status == "متصل" && !streaming) {
      return Colors.green; // connect without streaming
    }
    if (streaming) {
      return Colors.green; // active streaming
    }
    if (status == "متوقف مؤقتا") {
      return Colors.amberAccent; // paused
    }
    if (status == "غير متصل" || status == "فشل الإتصال" || status == "Connection error") {
      return Colors.redAccent; // not connected
    }
    return Colors.grey; // any other state
  }


  @override
  Widget build(BuildContext context) { //build main UI of camera screen
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Stack(
          children: [
           //full screen camera preview with aspect ratio preserved
            Positioned.fill(
              child: controller == null || !controller!.value.isInitialized
                  ? const Center(
                child: CircularProgressIndicator(color: Colors.white),
              )
                  : FittedBox(
                fit: BoxFit.cover, // يخلي الكاميرا تغطي الشاشة بدون تمدد
                child: SizedBox(
                  width: controller!.value.previewSize!.height,
                  height: controller!.value.previewSize!.width,
                  child: CameraPreview(controller!),
                ),
              ),
            ),

            Positioned( //at the top of the screen
              top: 20,
              left: 12,
              right: 12,
              child: Column(
                children: [
                  Row(
                    children: [
                      Icon(Icons.circle, size: 15, color: getStatusColor()), //connection status
                      SizedBox(width: 5),
                      Expanded(child: Text(" $status", style: TextStyle(color: Colors.white))),
                    ],
                  ),
                  SizedBox(height: 8),
                  Container(
                    padding: EdgeInsets.symmetric(vertical: 6, horizontal: 12),
                    decoration: BoxDecoration(color: Colors.black45, borderRadius: BorderRadius.circular(20)),
                    child: Directionality(
                      textDirection: TextDirection.rtl,
                      child: Text(
                        "الحرف الحالي: ${currentChar.isEmpty ? '...' : currentChar}",
                        style: TextStyle(color: Colors.white, fontSize: 18),
                      ),
                    ),
                  ),
                  SizedBox(height: 8),
                  Container(
                    padding: EdgeInsets.symmetric(vertical: 6, horizontal: 20),
                    decoration: BoxDecoration(color: Colors.black45, borderRadius: BorderRadius.circular(20)),
                    child: Directionality(
                      textDirection: TextDirection.rtl,
                      child: Text(
                        "الجملة: ${sentence.isEmpty ? '' : sentence}",
                        style: TextStyle(color: Colors.white, fontSize: 16),
                      ),
                    ),
                  ),
                ],
              ),
            ),


            Positioned( // Start & Stop button with dynamic color based on streaming status
              bottom: 130,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  width: 90,
                  height: 50,
                  child: ElevatedButton(
                    onPressed: onStop,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: streaming
                          ? Colors.green
                          : Color(0xFF858585),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                      padding: EdgeInsets.zero, // مهم حتى ما يكبر الزر زيادة
                    ),
                    child: Icon(
                      streaming ? Icons.pause : Icons.play_arrow,
                      size: 40,
                      color: streaming ? Colors.white : Colors.black87,
                    ),
                  ),
                ),
              ),
            ),



            Positioned( //buttons
              bottom: 50,
              left: 2,
              right: 2,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,// evenly spaces the buttons across the row
                children: [
                  cameraButton(Icons.delete_outline, onClear, Colors.redAccent), //clear
                  const SizedBox(width: 35),
                  cameraButton(Icons.remove, onRemove, Colors.orange),//remove
                  const SizedBox(width: 35),
                  cameraButton(Icons.add, onAdd, Colors.green),//add
                  const SizedBox(width: 35),
                  cameraButton(Icons.volume_up, onSpeak, Colors.blueAccent),//speak
                  const SizedBox(width: 35),
                  cameraButton(Icons.cameraswitch, switchCamera, Colors.white70),//switch
                ],
              ),
            ),
            Positioned(
              top: 10,
              right: 10,
              child: IconButton(
                icon: const Icon(Icons.info_outline, color: Colors.white70, size: 30),//dialog button
                onPressed: () => _showInfoDialog(context),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

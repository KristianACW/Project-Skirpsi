import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http_parser/http_parser.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:mime/mime.dart';
import 'result_page.dart';

void main() => runApp(const FoundationFitApp());

class FoundationFitApp extends StatelessWidget {
  const FoundationFitApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FoundationFit',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(fontFamily: 'Arial'),
      home: const CameraPage(),
    );
  }
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  File? _image;

  Future<void> _pickImage() async {
    final picked = await ImagePicker().pickImage(source: ImageSource.camera);
    if (picked != null) {
      setState(() => _image = File(picked.path));
    }
  }

  Future<void> _uploadAndProcess() async {
    if (_image == null) return;

    final uri = Uri.parse('http://192.168.0.5:5000/process'); // ganti IP Flask kamu
    final request = http.MultipartRequest('POST', uri);
    final mimeType = lookupMimeType(_image!.path)?.split('/');

    request.files.add(await http.MultipartFile.fromPath(
      'image',
      _image!.path,
      contentType: mimeType != null
          ? MediaType(mimeType[0], mimeType[1])
          : MediaType('image', 'jpeg'),
    ));

    try {
      final response = await request.send();
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final parts = responseBody.trim().split(';');

        if (parts.length == 3) {
          final algorithm = parts[0];
          final brand = parts[1];
          final colorHex = parts[2];
          print('Algorithm: $algorithm');
          print('Brand: $brand');
          print('Hex: $colorHex');

          if (!mounted) return;
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => ResultPage(
                algorithm: algorithm,
                brand: brand,
                colorHex: colorHex,
                imageFile: _image!,
              ),
            ),
          );
        } else {
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Format tidak sesuai: $responseBody')),
          );
        }
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Server Error: $responseBody')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Connection Error: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF9EAD8),
      body: SafeArea(
        child: Stack(
          children: [
            const Positioned(
              top: 20,
              left: 20,
              child: Text(
                'FoundationFit',
                style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold),
              ),
            ),
            Center(
              child: GestureDetector(
                onTap: _pickImage,
                child: _image == null
                    ? Stack(
                        alignment: Alignment.center,
                        children: [
                          const Icon(Icons.person_outline, size: 180),
                          Container(
                            width: 300,
                            height: 400,
                            decoration: BoxDecoration(
                              border: Border.all(color: Colors.orange, width: 10),
                              borderRadius: BorderRadius.circular(20),
                            ),
                          ),
                          const Positioned(
                            bottom: 40,
                            child: Text(
                              "Sesuaikan wajah di dalam kotak",
                              style: TextStyle(color: Colors.black, fontSize: 16),
                            ),
                          ),
                        ],
                      )
                    : Container(
                        width: 300,
                        height: 400,
                        decoration: BoxDecoration(
                          border: Border.all(color: Colors.orange, width: 10),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(20),
                          child: Image.file(_image!, fit: BoxFit.cover),
                        ),
                      ),
              ),
            ),
            Positioned(
              bottom: 30,
              left: 0,
              right: 0,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  IconButton(
                    icon: const Icon(Icons.delete, size: 30),
                    onPressed: () => setState(() => _image = null),
                  ),
                  const SizedBox(width: 20),
                  IconButton(
                    icon: const Icon(Icons.camera_alt, size: 30),
                    onPressed: _pickImage,
                  ),
                  const SizedBox(width: 20),
                  IconButton(
                    icon: const Icon(Icons.arrow_forward, size: 30),
                    onPressed: _uploadAndProcess,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

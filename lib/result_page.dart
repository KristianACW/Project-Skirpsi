import 'dart:io';
import 'package:flutter/material.dart';

class ResultPage extends StatelessWidget {
  final String algorithm;
  final String brand;
  final String colorHex;
  final File imageFile;

  const ResultPage({
    super.key,
    required this.algorithm,
    required this.brand,
    required this.colorHex,
    required this.imageFile,
  });

  /// Konversi HEX ke Color Flutter (dengan fallback warna abu-abu)
  Color getColorFromHex(String hex) {
    try {
      final buffer = StringBuffer();
      if (hex.length == 6 || hex.length == 7) buffer.write('ff'); // tambahkan alpha
      buffer.write(hex.replaceFirst('#', ''));
      return Color(int.parse(buffer.toString(), radix: 16));
    } catch (_) {
      return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    final resultColor = getColorFromHex(colorHex);

    return Scaffold(
      backgroundColor: const Color(0xFFF0E3C0), // Warna cream
      appBar: AppBar(
        title: const Text("Rekomendasi Foundation"),
        backgroundColor: Colors.orange,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // Gambar pengguna
            Center(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.file(
                  imageFile,
                  width: 200,
                  height: 280,
                  fit: BoxFit.cover,
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Algoritma
            const Text(
              "Algoritma:",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            Text(
              algorithm,
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),

            // Nama foundation
            const Text(
              "Foundation:",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            Text(
              brand,
              style: const TextStyle(fontSize: 16),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),

            // HEX code
            const Text(
              "Kode HEX:",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
            ),
            Text(
              colorHex,
              style: const TextStyle(fontSize: 15),
            ),
            const SizedBox(height: 20),

            // Preview warna
            Container(
              width: 60,
              height: 60,
              decoration: BoxDecoration(
                color: resultColor,
                shape: BoxShape.circle,
                border: Border.all(color: Colors.black),
              ),
            ),
            const SizedBox(height: 24),

            // Tombol kembali
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.orange,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
              ),
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.arrow_back),
              label: const Text("Kembali"),
            ),
          ],
        ),
      ),
    );
  }
}

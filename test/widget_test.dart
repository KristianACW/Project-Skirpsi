// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_skripsi/main.dart'; // ganti sesuai nama package kamu

void main() {
  testWidgets('Smoke test app runs', (WidgetTester tester) async {
    await tester.pumpWidget(CameraPage()); // ganti dengan nama class utama kamu
    expect(find.text('Halo'), findsOneWidget); // ubah sesuai teks yang muncul
  });
}

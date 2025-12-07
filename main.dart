import 'package:flutter/material.dart';
import 'Screens/first_page.dart';
import 'Screens/about_us_screen.dart';

void main() {
  runApp(
    // general style setting
    MaterialApp(
      debugShowCheckedModeBanner: false, // to hide the debug mark
      themeMode: ThemeMode.light,/////////////////////
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: Colors.white,        // background color of the pages
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          elevation: 0,
          foregroundColor: Colors.black,
        ),

        bottomNavigationBarTheme: const BottomNavigationBarThemeData(
          backgroundColor: Colors.white,            // the color of the bottom navigator
          selectedItemColor: Colors.blue,
          unselectedItemColor: Colors.grey,
        ),

        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
        ).copyWith(
          surface: Colors.white,
          surfaceContainerHighest: Colors.white,
          background: Colors.white,
        ),
      ),
      home: const HomeScreen(),
    ),
  );
}

class HomeScreen extends StatelessWidget {// begin of the welcome page
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [

              SizedBox(height: 80),
              // image setting
              Image.asset(
                'assets/images/Esmaa_Icon.png',
                height: 300,
                width: 300,
              ),

              // text setting and style
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                child: Text(
                  ' لكل اشارة معنى ولكل معنى صوت يُستحق أن يُسمع. '
                      'سيساعدك إسماع على ترجمة لغة الاشارة إلى نص وصوت عربي واضح، '
                      'حتى تكون الكلمات أقرب...والتواصل أسهل',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                      fontFamily: 'El_Messiri',
                      fontSize: 20,
                      fontWeight: FontWeight.bold
                  ),
                ),
              ),

              SizedBox(height: 40),
              // buttons settings
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // navigate about us screen
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => const AboutUsScreen()),
                      );
                    },
                    // button style
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.lightBlue[900],
                      padding: EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: Text(
                      "من نحن",
                      style: TextStyle(fontSize: 24, color: Colors.white),
                    ),
                  ),

                  SizedBox(width: 30),
                  // navigate to try now screen (home screen)
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => const TryScreen()),
                      );
                    },// button style
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.lightBlue[900],
                      padding: EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: Text(
                      "!جرب الان",
                      style: TextStyle(fontSize: 24, color: Colors.white),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
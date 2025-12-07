// first page
import 'package:flutter/material.dart';
import 'dictionary_screen.dart';
import 'camera_page.dart';

class TryScreen extends StatefulWidget {
  const TryScreen({super.key});

  @override
  State<TryScreen> createState() => _TryScreenState();
}

class _TryScreenState extends State<TryScreen> {
  int _selectedIndex = 0;

  // the three pages
  final List<Widget> _pages = [];

  @override
  void initState() {
    super.initState();

    _pages.addAll([
      // the first page
      Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            flex: 3,
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Image.asset(
                      "assets/labels/instructions.png",
                      width: 200,
                    ),
                  ),
                  const SizedBox(height: 25),

                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 20),
                    child: Text(
                      "هناك زران كما ترى في الشريط السفلي:",
                      textAlign: TextAlign.right,
                      style: TextStyle(
                        fontFamily: 'El_Messiri',
                        fontSize: 20,
                      ),
                    ),
                  ),

                  const SizedBox(height: 30),

                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(Icons.menu_book,
                            size: 28, color: Colors.black87),
                        const SizedBox(width: 10),
                        const Expanded(
                          child: Text(
                            "يوجّهك لقاموس الأحرف العربية بلغة الإشارة؛ يمكنك هناك البحث في القاموس ورؤية الإشارات بشكل أوضح.",
                            style: TextStyle(
                              fontFamily: 'El_Messiri',
                              fontSize: 20,
                              height: 1.6,
                              color: Colors.black87,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 24),

                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(Icons.camera_alt,
                            size: 28, color: Colors.black87),
                        const SizedBox(width: 10),
                        const Expanded(
                          child: Text(
                            "يوجّهك لبدء عملية الترجمة؛ يمكنك هناك ترجمة لغة الإشارة من حركات اليد إلى نص مكتوب ومسموع.",
                            style: TextStyle(
                              fontFamily: 'El_Messiri',
                              fontSize: 20,
                              height: 1.6,
                              color: Colors.black87,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 16),
                ],
              ),
            ),
          ),


          Align(
            alignment: Alignment.bottomCenter,
            child: Transform.scale(
              scale: 1.2, // to make the image bigger by 20%
              child: Image.asset(
                "assets/images/homePage_image.png",
              ),
            ),
          )


        ],
      ),

      // dictionary screen
      const DictionaryScreen(),

      Directionality(
        textDirection: TextDirection.ltr,
        child: const CameraPage(),
      ),
    ]);
  }

  // used to know which page is selected and navigate to it
  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        backgroundColor: Colors.white,

        appBar: AppBar(
          backgroundColor: Colors.white,
          elevation: 0,
          automaticallyImplyLeading: false,
          leading: IconButton(
            icon: const Icon(Icons.arrow_back, color: Colors.black),
            onPressed: () => Navigator.pop(context),
          ),
        ),

        body: IndexedStack(
          index: _selectedIndex, // to know what page is selected
          children: _pages, // Dictionary , camera etc..
        ),
        bottomNavigationBar: BottomNavigationBar(
          backgroundColor: Colors.white,
          currentIndex: _selectedIndex,
          onTap: _onItemTapped,
          selectedItemColor: Colors.blue,
          unselectedItemColor: Colors.grey,
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.home),
              label: "الصفحة الرئيسية",
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.menu_book),
              label: "القاموس",
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.camera_alt),
              label: "الكاميرا",
            ),
          ],
        ),
      ),
    );
  }
}
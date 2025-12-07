//about us
import 'package:flutter/material.dart';

class AboutUsScreen extends StatelessWidget {
  const AboutUsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,

      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        automaticallyImplyLeading: false,
        actions: [
          IconButton(
            icon: const Icon(Icons.arrow_forward, color: Colors.black),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),

      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [

          const SizedBox(height: 16),

          // title
          Center(
            child: Image.asset(
              'assets/labels/aboutUs.png',
              width: 200,
            ),
          ),

          const SizedBox(height: 18),


          Expanded(
            flex: 3,
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              child: const Padding(
                padding: EdgeInsets.symmetric(horizontal: 20),
                child: Directionality(
                  textDirection: TextDirection.rtl,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [

                      Text(
                        "نحن طالبات كلية علوم الحاسب والمعلومات من جامعة الإمام محمد بن سعود الإسلامية..",
                        textAlign: TextAlign.right,
                        style: TextStyle(
                            fontSize: 17,
                            height: 1.7,
                            fontFamily: 'El_Messiri'),
                      ),
                      SizedBox(height: 6),

                      Text(
                        "جاءت فكرة مشروعنا ليكون جسرًا لإيصال المشاعر قبل الكلمات، ويمنح لغة الإشارة فرصة لتُسمع كما تُرى.",
                        textAlign: TextAlign.right,
                        style: TextStyle(
                            fontSize: 17,
                            height: 1.7,
                            fontFamily: 'El_Messiri'),
                      ),
                      SizedBox(height: 6),

                      Text(
                        "هدفنا أن نُمكّن فئة الصم وضعاف السمع من التعبير بكل يسر وسهولة، وأن نساهم في جعل مجتمعنا أكثر احتواءً وتواصلاً بلا حواجز.",
                        textAlign: TextAlign.right,
                        style: TextStyle(
                            fontSize: 17,
                            height: 1.7,
                            fontFamily: 'El_Messiri'),
                      ),
                      SizedBox(height: 6),

                      Text(
                        "تطبيقنا يحوّل الإشارات إلى أحرف، ثم كلمات، ثم إلى صوت عربي مفهوم للجميع؛ لأن كل إشارة تحمل رسالة، وكل رسالة تستحق أن تُقال.",
                        textAlign: TextAlign.right,
                        style: TextStyle(
                            fontSize: 17,
                            height: 1.7,
                            fontFamily: 'El_Messiri'),
                      ),
                      SizedBox(height: 6),

                      Text(
                        "نحن لا نبني تطبيقًا فقط، بل نفتح بابًا جديدًا للتواصل الإنساني. وبكل خطوة، نقترب من يوم تصبح فيه التكنولوجيا صوتًا لمن لم يُسمع من قبل.",
                        textAlign: TextAlign.right,
                        style: TextStyle(
                            fontSize: 17,
                            height: 1.7,
                            fontFamily: 'El_Messiri'),
                      ),

                      SizedBox(height: 16),
                    ],
                  ),
                ),
              ),
            ),
          ),

          //  الصورة أسفل الشاشة
          Expanded(
            flex: 1,
            child: Image.asset(
              "assets/images/AboutUs_image.jpeg",
              width: double.infinity,
              fit: BoxFit.cover,
            ),
          ),
        ],
      ),
    );
  }
}
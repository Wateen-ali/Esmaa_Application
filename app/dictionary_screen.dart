import 'package:flutter/material.dart';

class DictionaryScreen extends StatefulWidget {
  const DictionaryScreen({super.key});
  @override
  State<DictionaryScreen> createState() => _DictionaryScreenState();
}

class _DictionaryScreenState extends State<DictionaryScreen> {
  final TextEditingController _searchCtrl = TextEditingController();
  String _q = ''; // value for the entered text

  // list of the arabic sign letters
  final List<SignEntry> _all = const [
    SignEntry(letter: 'أ',  label: 'ألف',           code: 'ALIF',      assetPath: 'assets/signs/ar/أ.png'),
    SignEntry(letter: 'ال', label: 'الـ (تعريف)',   code: 'AL',        assetPath: 'assets/signs/ar/ال.png'),
    SignEntry(letter: 'ب',  label: 'باء',           code: 'BA',        assetPath: 'assets/signs/ar/ب.png'),
    SignEntry(letter: 'د',  label: 'دال',           code: 'DAL',       assetPath: 'assets/signs/ar/د.png'),
    SignEntry(letter: 'ض',  label: 'ضاد',           code: 'DAD',       assetPath: 'assets/signs/ar/ض.png'),
    SignEntry(letter: 'ظ',  label: 'ظاء',           code: 'ZHA',       assetPath: 'assets/signs/ar/ظ.png'),
    SignEntry(letter: 'ع',  label: 'عين',           code: 'AIN',       assetPath: 'assets/signs/ar/ع.png'),
    SignEntry(letter: 'ف',  label: 'فاء',           code: 'FA',        assetPath: 'assets/signs/ar/ف.png'),
    SignEntry(letter: 'ة',  label: 'تاء مربوطة',    code: 'TA_MARBUTA',assetPath: 'assets/signs/ar/ة.png'),
    SignEntry(letter: 'ت',  label: 'تاء',           code: 'TA',        assetPath: 'assets/signs/ar/ت.png'),
    SignEntry(letter: 'ث',  label: 'ثاء',           code: 'THA',       assetPath: 'assets/signs/ar/ث.png'),
    SignEntry(letter: 'ج',  label: 'جيم',           code: 'JIM',       assetPath: 'assets/signs/ar/ج.png'),
    SignEntry(letter: 'ح',  label: 'حاء',           code: 'HAH',       assetPath: 'assets/signs/ar/ح.png'),
    SignEntry(letter: 'خ',  label: 'خاء',           code: 'KHA',       assetPath: 'assets/signs/ar/خ.png'),
    SignEntry(letter: 'ذ',  label: 'ذال',           code: 'THAL',      assetPath: 'assets/signs/ar/ذ.png'),
    SignEntry(letter: 'ر',  label: 'راء',           code: 'RA',        assetPath: 'assets/signs/ar/ر.png'),
    SignEntry(letter: 'ز',  label: 'زاي',           code: 'ZAY',       assetPath: 'assets/signs/ar/ز.png'),
    SignEntry(letter: 'س',  label: 'سين',           code: 'SIN',       assetPath: 'assets/signs/ar/س.png'),
    SignEntry(letter: 'ش',  label: 'شين',           code: 'SHIN',      assetPath: 'assets/signs/ar/ش.png'),
    SignEntry(letter: 'ص',  label: 'صاد',           code: 'SAD',       assetPath: 'assets/signs/ar/ص.png'),
    SignEntry(letter: 'ط',  label: 'طاء',           code: 'TTA',       assetPath: 'assets/signs/ar/ط.png'),
    SignEntry(letter: 'غ',  label: 'غين',           code: 'GHAYN',     assetPath: 'assets/signs/ar/غ.png'),
    SignEntry(letter: 'ق',  label: 'قاف',           code: 'QAF',       assetPath: 'assets/signs/ar/ق.png'),
    SignEntry(letter: 'ك',  label: 'كاف',           code: 'KAF',       assetPath: 'assets/signs/ar/ك.png'),
    SignEntry(letter: 'ل',  label: 'لام',           code: 'LAM',       assetPath: 'assets/signs/ar/ل.png'),
    SignEntry(letter: 'لا', label: 'لا',            code: 'LAM_ALEF',  assetPath: 'assets/signs/ar/لا.png'),
    SignEntry(letter: 'م',  label: 'ميم',           code: 'MIM',       assetPath: 'assets/signs/ar/م.png'),
    SignEntry(letter: ' ',  label: 'مسافة',         code: 'SPACE',     assetPath: 'assets/signs/ar/مسافة.png'),
    SignEntry(letter: 'ن',  label: 'نون',           code: 'NUN',       assetPath: 'assets/signs/ar/ن.png'),
    SignEntry(letter: 'هـ', label: 'هاء',           code: 'HA',        assetPath: 'assets/signs/ar/هـ.png'),
    SignEntry(letter: 'و',  label: 'واو',           code: 'WAW',       assetPath: 'assets/signs/ar/و.png'),
    SignEntry(letter: 'ي',  label: 'ياء',           code: 'YA',        assetPath: 'assets/signs/ar/ي.png'),
    SignEntry(letter: 'ئ',  label: 'ياء همزة',      code: 'YA_HAMZA',  assetPath: 'assets/signs/ar/ئ.png'),
  ];


  @override
  void dispose() { _searchCtrl.dispose(); super.dispose(); } // to clear the cash when existing the page

  @override
  // when the search par is empty return all the letter other ways the written letter will be shown
  Widget build(BuildContext context) {
    final filtered = _all.where((e) {
      if (_q.trim().isEmpty) return true;
      final q = _normalize(_q);
      return _normalize(e.letter).contains(q) ||
          _normalize(e.label).contains(q)  ||
          _normalize(e.code).contains(q);
    }).toList()..sort((a,b)=>a.letter.compareTo(b.letter));

    return Directionality(
      textDirection: TextDirection.rtl,
      child: Scaffold(
        backgroundColor: Colors.white,
        body: Scrollbar(
          thumbVisibility: true, // to show the scroll bar
          child: CustomScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            slivers: [
              // شريط البحث
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16,16,16,8),
                  child: TextField(
                    controller: _searchCtrl,
                    onChanged: (v)=>setState(()=>_q=v),
                    textInputAction: TextInputAction.search,
                    decoration: InputDecoration(
                      hintText: 'ابحث: أ، ب، عين، ALIF…',
                      prefixIcon: const Icon(Icons.search),
                      suffixIcon: _q.isEmpty ? null : IconButton(
                        onPressed: () { _searchCtrl.clear(); setState(()=>_q=''); },
                        icon: const Icon(Icons.clear),
                      ),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(14)),
                    ),
                  ),
                ),
              ),

              // if there is no result
              if (filtered.isEmpty)
                const SliverFillRemaining(
                  hasScrollBody: false,
                  child: Center(child: Text('لا يوجد نتيجة مطابقة')),
                )
              else // if there is a result
                SliverPadding(
                  padding: const EdgeInsets.fromLTRB(16,8,16,16),
                  // the result letter in grid
                  sliver: SliverGrid(
                    gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 3, crossAxisSpacing: 12, mainAxisSpacing: 12, childAspectRatio: .9),
                    delegate: SliverChildBuilderDelegate(
                          (context, i) => _SignCard(item: filtered[i]),
                      childCount: filtered.length,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
// making the search easier
  String _normalize(String s) => s
      .replaceAll('أ','ا').replaceAll('إ','ا').replaceAll('آ','ا')
      .replaceAll('ى','ي').replaceAll('ة','ه').replaceAll('ـ','')
      .toLowerCase();
}

class SignEntry {
  final String letter, label, code, assetPath;
  const SignEntry({required this.letter, required this.label, required this.code, required this.assetPath});
}
// sign card for each letter to show the name and image of the letter
class _SignCard extends StatelessWidget {
  final SignEntry item; const _SignCard({required this.item});
  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(16),
      onTap: () => showDialog( // when tapping on the letter it will make the image bigger
        context: context,
        builder: (_) => Dialog(
          clipBehavior: Clip.antiAlias,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: AspectRatio(aspectRatio: 1, child: Image.asset(item.assetPath, fit: BoxFit.contain)),
        ),
      ),
      // the card style setting
      child: Ink(
        decoration: BoxDecoration(
          color:  Theme.of(context).colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(.06), blurRadius: 8, offset: const Offset(0,2))],
        ),
        child: Column(
            children: [
        Expanded(
        child: Padding(
        padding: const EdgeInsets.all(8),
        child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.asset(
                item.assetPath,
                fit: BoxFit.cover,
                errorBuilder: (_, _, _) => const Center(child: Icon(Icons.broken_image_outlined, size: 32)),
      ),
    ),
    ),
    ),
    Padding(
    padding: const EdgeInsets.fromLTRB(8,0,8,10),
    child: Text(item.label, maxLines: 1, overflow: TextOverflow.ellipsis,
    style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
    ),
    ],
    ),
    ),
    );
  }
}

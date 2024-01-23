
# Sequence to Sequence Yöntemiyle Türkçe-İngilizce Çeviri

Geliştirdiğim bu proje, Bahdanau Mekanizması ile güçlendirilmiş Sequence to Sequence algoritmalarını kullanarak Türkçe-İngilizce çeviri konusunda bir çözüm sunmaktadır.

Sequence to Sequence yöntemi, özellikle doğal dil işleme ve makine çevirisi gibi uygulamalarda kullanılan bir derin öğrenme modeli türüdür. Seq2Seq modelleri, girişteki bir diziyi alıp çıkıştaki başka bir dizi üretme yeteneğine sahiptir. Genellikle bir dilin cümlelerini başka bir dilin cümlelerine çevirme gibi çeşitli sekans eşleme görevlerinde kullanılır.

Bahdanau Mekanizması, özellikle dil çevirisi gibi uzun ve karmaşık dizilerin işlendiği Seq2Seq modellerde kullanılan bir dikkat mekanizmasıdır. Bu mekanizma, modelin çıkışını oluştururken her bir giriş elemanına ağırlıklarını dinamik olarak atamasını sağlar. Bu sayede model, özellikle uzun giriş dizileriyle daha iyi başa çıkabilir ve önemli bilgileri odaklanarak kullanabilir.

## Dosya Yükleme

Projeyi yükleyin

```bash
    git clone https://github.com/VuralBayrakli/seq2seq.git
```

Gerekli kütüphaneleri yükleyin
```bash
    pip install -r requirements.txt
```

## Örnekler

![App Screenshot](https://github.com/VuralBayrakli/seq2seq/blob/master/ss/ss1.jpg)

![App Screenshot](https://github.com/VuralBayrakli/seq2seq/blob/master/ss/ss2.jpg)

![App Screenshot](https://github.com/VuralBayrakli/seq2seq/blob/master/ss/ss3.jpg)

import matplotlib.pyplot as plt
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

# دالة لتصحيح النصوص العربية لظهر متصلة وصحيحة في الرسم
def ar(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# بيانات دورات التحقق المتقاطع (5-Fold Cross-Validation)
folds = [ar('الدورة الأولى'), ar('الدورة الثانية'), ar('الدورة الثالثة'), ar('الدورة الرابعة'), ar('الدورة الخامسة')]
scores = cv_scores  # النتائج الموجودة لديك مسبقاً

# إعداد تصميم رسمة إبداعي وعصري
plt.figure(figsize=(11, 6))
colors = ['#3498DB', '#1ABC9C', '#9B59B6', '#E67E22', '#2ECC71']

# رسم الأعمدة بشكل أنيق
bars = plt.bar(folds, scores, color=colors, width=0.55, edgecolor='none', alpha=0.9, zorder=3)

# إضافة خط أفقي يوضح متوسط الدقة (Mean Accuracy)
mean_score = np.mean(scores)
plt.axhline(y=mean_score, color='#E74C3C', linestyle='--', linewidth=2.2, zorder=4, 
            label=ar(f'متوسط الدقة العام: {mean_score:.4f}'))

# العناوين باللغة العربية وبشكل رسمي ومميز
plt.title(ar('تقييم استقرار نموذج Random Forest عبر التحقق المتقاطع (5-Fold CV)'), fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
plt.xlabel(ar('دورات التحقق (Folds)'), fontsize=12, fontweight='bold', labelpad=10, color='#34495E')
plt.ylabel(ar('معدل الدقة (Accuracy)'), fontsize=12, fontweight='bold', labelpad=10, color='#34495E')

# تنسيق المخطط وخلفيته
plt.ylim(0.80, 1.02)
plt.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)

# إضافة القيم الدقيقة فوق كل عمود بشكل جذاب داخل مربعات شفافة
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#BDC3C7', alpha=0.85))

# إضافة مفتاح الرسم (Legend)
plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#BDC3C7', fontsize=11)

plt.tight_layout()
plt.show()

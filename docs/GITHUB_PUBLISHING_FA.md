# راهنمای انتشار LiBCaT روی GitHub

## ۱. تست روی سیستم خودت

در اوبونتو، ZIP را از حالت فشرده خارج کن و در پوشه پروژه این دستورها را اجرا کن:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_env.py
python -m unittest discover -s tests -v
python scripts/smoke_test.py
streamlit run app.py
```

در رابط برنامه، ابتدا **Lexical (offline)** را انتخاب کن. داده پیش‌فرض کاملاً **ساختگی** است. نصب وابستگی‌های اصلی به اینترنت نیاز دارد، اما اجرای BM25 به دانلود مدل نیازی ندارد.

برای مدل‌های سنگین‌تر:

```bash
python -m pip install -r requirements-dense.txt
streamlit run app.py
```

در رابط، **Hybrid** یا **Biomedical rerank** را انتخاب کن. مدل‌ها هنگام نخستین اجرا دانلود می‌شوند. روی CPU ممکن است کند باشند. برای اطمینان از اجرای واقعی مدل‌های اختیاری (نه صرفاً باز شدن اپ)، این تست جدا را هم اجرا کن:

```bash
python scripts/verify_optional_models.py --mode biomedical
```

این تست به اینترنت و دانلود وزن مدل‌ها نیاز دارد و در CI سبک اجرا نمی‌شود. قبل از آزمایش آن نمی‌توان گفت MedCPT واقعاً روی سیستم اجرا شده است.

## ۲. دیتای TREC حجیم

شش آرشیو ۲۰۲۳ را به همان نام اصلی در `data/trec/` بگذار. این پوشه در `.gitignore` است؛ در GitHub آپلود نمی‌شود. اگر این شش فایل را داری:

```bash
python scripts/download_trec_data.py
python scripts/parse_trec_corpus.py
python scripts/run_trec_eval.py --mode lexical
```

دستور اول فقط فایل‌های کوچک topics و qrels را در صورت نیاز دریافت می‌کند. برای دریافت خود آرشیوها از منبع بالادستی: `python scripts/download_trec_data.py --corpus`. اگر آدرس‌های بالادستی محدود یا در دسترس نبودند، فایل‌ها را دستی در مسیر موردنظر قرار بده. برای انتشار مجدد مجموعه‌داده، ابتدا مجوز صاحب داده را بررسی کن.

## ۳. انتشار مخزن روی اکانت شخصی

اول در GitHub یک مخزن عمومی **خالی** با نام `LiBCaT` بساز؛ هنگام ساخت، README و License جدید انتخاب نکن، چون فایل‌ها اینجا هستند.

```bash
git init -b main
git add .
git status --short
git commit -m "Initial public release of LiBCaT"
git remote add origin https://github.com/YOUR_USERNAME/LiBCaT.git
git push -u origin main
```

`YOUR_USERNAME` را با نام کاربری خودت عوض کن. برای push باید GitHub authentication معتبر داشته باشی (SSH یا HTTPS credential). پیش از commit بررسی کن که داده‌های ۲ تا ۳ گیگابایتی و فایل‌های `.env` در `git status` نباشند.

اگر Git از قبل در پروژه راه‌اندازی شده، `git init` را تکرار نکن؛ فقط وضعیت remote و branch را بررسی کن.

## ۴. افزودن دوستت و همکاری روی پروژه

در مخزن GitHub به **Settings → Collaborators and teams** (یا گزینه مشابه) برو و دوستت را با نام کاربری GitHub دعوت کن. بعد از پذیرفتن دعوت، بهتر است برای هر تغییر یک Branch و Pull Request ایجاد کنید. در **Settings → Branches / Rulesets** برای شاخه `main` الزامی‌کردن PR و عبور از CI را فعال کن.

فایل‌های `CONTRIBUTING.md` و قالب Pull Request آماده هستند. در GitHub، بخش **Actions** اجرای تست‌ها را روی push و PR نمایش می‌دهد.

## ۵. قبل از عمومی کردن

متن README، ادعاهای آزمایشگاهی، مجوز MIT، مالکیت کد و هرگونه حق انتشار داده/مدل را بازبینی کن. نام نویسندگان/دوستان را می‌توان بعداً در README ثبت کرد؛ مشارکت‌هایی که از طریق commit و Pull Request ثبت می‌شوند نیز در GitHub قابل پیگیری‌اند.

## ۶. مسیر ادامه توسعه بعد از انتشار

بعد از انتشار، توسعه‌دهنده می‌تواند از بخش **Developer workflow** در
[README](../README.md#developer-workflow) شروع کند: Clone یا Fork، نصب توسعه‌ای،
اجرای تست بدون دیتای چندگیگابایتی، شناخت پوشه‌های پروژه، ایجاد Branch، ثبت تست و
ساخت Pull Request. معیارهای مشارکت در [CONTRIBUTING](../CONTRIBUTING.md) آمده‌اند.

برای دوستت، بعد از دعوت و تأیید دسترسی، بهتر است اولین تغییر را در یک
`feat/...` Branch و از طریق PR انجام دهد، نه با Push مستقیم به `main`.

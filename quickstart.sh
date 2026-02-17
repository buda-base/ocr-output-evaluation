#!/bin/bash
# Quick Start Script - Run this first!

echo "🚀 OCR Output Evaluation - Quick Start"
echo "======================================"
echo ""

# Check if we're in the right directory
if [ ! -f "analyze_confidence.py" ]; then
    echo "❌ Error: Please run this script from the project directory"
    exit 1
fi

echo "Step 1/5: Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "⚠️  Some dependencies may have failed. Run: pip install -r requirements.txt"
fi
echo ""

echo "Step 2/5: Creating output directories..."
mkdir -p output plots
echo "✅ Directories created"
echo ""

echo "Step 3/5: Testing database connection..."
python -c "from db_queries import get_all_volumes; gb, gv = get_all_volumes(); print(f'✅ Database OK: {len(gb)} GB volumes, {len(gv)} GV volumes')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Database connection issue. Check .env file"
fi
echo ""

echo "Step 4/5: Testing S3 access..."
python -c "import pandas as pd; from db_queries import get_all_volumes; from config import S3_GB_PATH_TEMPLATE, S3_BUCKET; gb, gv = get_all_volumes(); vol = gb[0] if gb else None; df = pd.read_parquet(S3_GB_PATH_TEMPLATE.format(bucket=S3_BUCKET, w_id=vol['w_id'], i_id=vol['i_id'], i_version=vol['i_version'])) if vol else None; print('✅ S3 access OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  S3 access issue. Check AWS credentials"
fi
echo ""

echo "Step 5/5: Running test analysis (5 volumes)..."
python analyze_confidence.py --limit 5
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test analysis complete!"
    echo ""
    echo "📊 Check your results:"
    echo "   ls -lh output/"
    echo ""
    echo "🔍 Explore results:"
    echo "   python explore_stats.py --summary"
    echo ""
    echo "🚀 Ready for full analysis:"
    echo "   python analyze_confidence.py"
    echo ""
    echo "📖 Documentation:"
    echo "   - QUICKSTART.md"
    echo "   - README.md"
    echo "   - SUMMARY.md"
    echo ""
else
    echo "❌ Test failed. Check the error messages above."
    echo "   Run: python test_setup.py for detailed diagnostics"
fi

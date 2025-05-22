#include "gdal.h"
#include "gdal_priv.h"
#include "gdal_utils.h"
#include "ogrsf_frmts.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

int main()
{
    CPLSetConfigOption("GDAL_NUM_THREADS", "1");

    GDALAllRegister();

    auto *d = GetGDALDriverManager()->GetDriverByName("Memory");
    if (d == nullptr)
        return -1;

    auto *ds = d->Create("", 0, 0, 0, GDT_Unknown, nullptr);
    auto *lyr = ds->CreateLayer("test");

    auto *feat = OGRFeature::CreateFeature(lyr->GetLayerDefn());

    {
        OGRGeometry *geo = nullptr;
        OGRGeometryFactory::createFromWkt("POINT(0 0 0)", nullptr, &geo);
        feat->SetGeometryDirectly(geo);
    }

    for (size_t i = 0; i < 100000; ++i)
        if (lyr->CreateFeature(feat->Clone()) != OGRERR_NONE)
        {
            printf("failed to create feat\n");
            return -1;
        }

    const char *options[] = {
        "-outsize", "50",      "50",    //
        "-txe",     "-0.25",   "1.25",  //
        "-tye",     "-0.25",   "1.25",  //
        "-of",      "MEM",              //
        "-a",       "invdist",          //
        nullptr,
    };

    auto *opts = GDALGridOptionsNew(const_cast<char **>(options), nullptr);

    clock_t start = clock();

    auto *out = GDALDataset::FromHandle(
        GDALGrid("", GDALDataset::ToHandle(ds), opts, nullptr));

    clock_t end = clock();

    if (out == nullptr)
        return -1;

    printf("(%dx%d) %f\n", out->GetRasterXSize(), out->GetRasterYSize(),
           (end - start) * 1.0 / CLOCKS_PER_SEC);
}

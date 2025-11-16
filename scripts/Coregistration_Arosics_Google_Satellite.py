
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Coregister WorldView or other imagery to a Google Satellite reference using AROSICS.")
    parser.add_argument('--reference', required=True, help='Path to reference image (e.g., Google Satellite)')
    parser.add_argument('--target', required=True, help='Path to target image to be coregistered')
    parser.add_argument('--output', required=True, help='Output path for coregistered image')
    parser.add_argument('--grid_res', type=float, default=1, help='Grid resolution for coregistration (default: 1)')
    parser.add_argument('--window_size', type=int, nargs=2, default=[300, 300], help='Window size (default: 300 300)')
    parser.add_argument('--max_shift', type=float, default=15, help='Maximum shift (default: 15)')
    parser.add_argument('--max_iter', type=int, default=25, help='Maximum iterations (default: 25)')
    parser.add_argument('--resamp_alg_calc', default='nearest', help='Resampling algorithm for calculation (default: nearest)')
    parser.add_argument('--resamp_alg_deshift', default='nearest', help='Resampling algorithm for deshift (default: nearest)')
    parser.add_argument('--nodata', type=int, nargs=2, default=[0,0], help='Nodata values (default: 0 0)')
    parser.add_argument('--fmt_out', default='GTIFF', help='Output format (default: GTIFF)')
    parser.add_argument('--max_points', type=int, default=10000, help='Maximum points (default: 10000)')
    parser.add_argument('--align_grids', action='store_true', help='Align grids (default: False)')
    parser.add_argument('--CPUs', type=int, default=16, help='Number of CPUs (default: 16)')
    parser.add_argument('--r_b4match', type=int, default=1, help='Reference band for matching (default: 1)')
    parser.add_argument('--s_b4match', type=int, default=5, help='Target band for matching (default: 5)')
    parser.add_argument('--save_tiepoints', action='store_true', help='Save tiepoints and parameters as GeoJSON/CSV/JSON')
    return parser.parse_args()

def main():
    args = parse_args()
    from arosics import COREG_LOCAL
    import rioxarray as rxr
    import geopandas as gpd
    from shapely.geometry import Point
    import json
    import os

    kwargs = {
        'grid_res': args.grid_res,
        'window_size': tuple(args.window_size),
        'path_out': args.output,
        'max_shift': args.max_shift,
        'max_iter': args.max_iter,
        'resamp_alg_calc': args.resamp_alg_calc,
        'resamp_alg_deshift': args.resamp_alg_deshift,
        'nodata': args.nodata,
        'v': True,
        'fmt_out': args.fmt_out,
        'max_points': args.max_points,
        'align_grids': args.align_grids,
        'CPUs': args.CPUs,
        'r_b4match': args.r_b4match,
        's_b4match': args.s_b4match,
    }

    print('Coregistration parameters:', kwargs)
    print('Reference:', args.reference)
    print('Target:', args.target)
    print('Output:', args.output)

    CRL = COREG_LOCAL(args.reference, args.target, **kwargs)
    CRL.correct_shifts()
    print("Coregistration completed.")

    if args.save_tiepoints:
        # Save tiepoints and parameters
        ref_image = rxr.open_rasterio(CRL.params['im_ref'])
        ref_crs = ref_image.rio.crs
        output_prefix = os.path.splitext(args.output)[0]
        shp_tiepoints = output_prefix + '_CRL_CoregPoints.geojson'
        tiepoints_df = CRL.CoRegPoints_table
        tiepoints_df['geometry'] = tiepoints_df['geometry'].apply(lambda geom: Point(geom.x, geom.y))
        tiepoints_gdf = gpd.GeoDataFrame(tiepoints_df, geometry='geometry', crs=ref_crs)
        tiepoints_gdf = tiepoints_gdf.select_dtypes(exclude=['object', 'complex'])
        tiepoints_gdf.to_file(shp_tiepoints, driver='GeoJSON')
        print(f"Tie points saved in: {shp_tiepoints}")
        coreg_points_filepath = output_prefix + '_CRL_CoregPoints_table.csv'
        coreg_params_filepath = output_prefix + '_CRL_CoregParams.json'
        CRL.CoRegPoints_table.to_csv(coreg_points_filepath, index=False)
        with open(coreg_params_filepath, 'w') as f:
            json.dump(CRL.params, f, indent=4)
        print(f"Tie points table saved in: {coreg_points_filepath}")
        print(f"Parameters saved in: {coreg_params_filepath}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run the DBSCAN clustering analysis and generate the final report
"""

import os
import shutil
import subprocess
import time
from datetime import datetime

def create_report_directory():
    """Create a directory for the final report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"high_silhouette_dbscan_report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    print(f"Created report directory: {report_dir}")
    return report_dir

def run_visualization_script():
    """Run the visualization script"""
    print("\nRunning visualization script...")
    result = subprocess.run(["python3", "visualize_dbscan_clusters.py"], 
                           capture_output=True, text=True)
    if result.returncode == 0:
        print("Visualization completed successfully!")
    else:
        print("Error running visualization script:")
        print(result.stderr)
    return "dbscan_visualizations"  # Default output directory from the script

def copy_markdown_report(report_dir):
    """Copy the markdown report to the report directory"""
    print("\nCopying markdown report...")
    shutil.copy("high_silhouette_dbscan_report.md", 
                os.path.join(report_dir, "high_silhouette_dbscan_report.md"))
    
def copy_visualizations(vis_dir, report_dir):
    """Copy the visualizations to the report directory"""
    print("\nCopying visualizations...")
    # Create visualizations subdirectory
    vis_output_dir = os.path.join(report_dir, "visualizations")
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # Copy all files from visualization directory
    for file in os.listdir(vis_dir):
        src = os.path.join(vis_dir, file)
        dst = os.path.join(vis_output_dir, file)
        shutil.copy(src, dst)
    
def copy_best_cluster_stats(report_dir):
    """Copy the best cluster stats file to the report directory"""
    print("\nCopying best cluster statistics...")
    stats_file = "improved_clustering_20250425_131958/best_cluster_stats.txt"
    if os.path.exists(stats_file):
        shutil.copy(stats_file, 
                    os.path.join(report_dir, "best_cluster_stats.txt"))
    else:
        print(f"Warning: Could not find {stats_file}")

def create_html_report(report_dir):
    """Create an HTML version of the report"""
    # Check if pandoc is installed
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        has_pandoc = True
    except (subprocess.SubprocessError, FileNotFoundError):
        has_pandoc = False
        print("Warning: Pandoc not found. Skipping HTML report generation.")
        return
        
    if has_pandoc:
        print("\nGenerating HTML report...")
        md_file = os.path.join(report_dir, "high_silhouette_dbscan_report.md")
        html_file = os.path.join(report_dir, "high_silhouette_dbscan_report.html")
        
        # Add CSS for better styling
        css = """
        <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
        }
        </style>
        """
        
        # Run pandoc to convert markdown to HTML
        result = subprocess.run([
            "pandoc", 
            md_file, 
            "-o", html_file, 
            "--standalone",
            "--metadata", "title=High Silhouette Score Customer Clustering Analysis Report"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Add the CSS to the generated HTML
            with open(html_file, 'r') as f:
                html_content = f.read()
            
            # Insert CSS after the <head> tag
            if "<head>" in html_content:
                html_content = html_content.replace("<head>", f"<head>{css}")
                
                # Add links to visualizations
                vis_dir = os.path.join("visualizations")
                visualization_html = "<h2>Interactive Visualizations</h2>\n<div>"
                for file in os.listdir(os.path.join(report_dir, vis_dir)):
                    if file.endswith(".png"):
                        img_path = os.path.join(vis_dir, file)
                        img_title = file.replace("_", " ").replace(".png", "").title()
                        visualization_html += f"""
                        <div style="margin-bottom: 40px;">
                            <h3>{img_title}</h3>
                            <img src="{img_path}" alt="{img_title}" style="max-width: 800px;">
                        </div>
                        """
                visualization_html += "</div>"
                
                # Add visualizations section before the closing body tag
                html_content = html_content.replace("</body>", f"{visualization_html}</body>")
                
                with open(html_file, 'w') as f:
                    f.write(html_content)
                
                print(f"HTML report created: {html_file}")
            else:
                print("Warning: Could not add CSS to HTML report")
        else:
            print("Error generating HTML report:")
            print(result.stderr)

def create_report_index(report_dir):
    """Create an index.html file for the report"""
    index_path = os.path.join(report_dir, "index.html")
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DBSCAN Customer Clustering Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #2c3e50;
        }
        .report-section {
            margin-bottom: 30px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .viz-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .viz-item {
            text-align: center;
        }
        .viz-item img {
            max-width: 100%;
            height: auto;
            border: 1px solid #eee;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>High Silhouette Score DBSCAN Customer Clustering</h1>
    
    <div class="report-section">
        <h2>Report Documents</h2>
        <ul>
"""
    
    # Add links to report files
    if os.path.exists(os.path.join(report_dir, "high_silhouette_dbscan_report.html")):
        html_content += '            <li><a href="high_silhouette_dbscan_report.html">Complete HTML Report</a></li>\n'
    
    html_content += '            <li><a href="high_silhouette_dbscan_report.md">Markdown Report</a></li>\n'
    
    if os.path.exists(os.path.join(report_dir, "best_cluster_stats.txt")):
        html_content += '            <li><a href="best_cluster_stats.txt">Detailed Cluster Statistics</a></li>\n'
    
    if os.path.exists(os.path.join(report_dir, "visualizations/cluster_summary.csv")):
        html_content += '            <li><a href="visualizations/cluster_summary.csv">Cluster Summary (CSV)</a></li>\n'
    
    if os.path.exists(os.path.join(report_dir, "visualizations/detailed_report.txt")):
        html_content += '            <li><a href="visualizations/detailed_report.txt">Detailed Analysis Report</a></li>\n'
    
    html_content += """
        </ul>
    </div>
    
    <div class="report-section">
        <h2>Key Visualizations</h2>
        <div class="viz-gallery">
"""
    
    # Add visualization gallery
    vis_dir = os.path.join(report_dir, "visualizations")
    if os.path.exists(vis_dir):
        for file in os.listdir(vis_dir):
            if file.endswith(".png"):
                img_path = os.path.join("visualizations", file)
                img_title = file.replace("_", " ").replace(".png", "").title()
                html_content += f"""
            <div class="viz-item">
                <a href="{img_path}">
                    <img src="{img_path}" alt="{img_title}">
                    <p>{img_title}</p>
                </a>
            </div>"""
    
    html_content += """
        </div>
    </div>
    
    <footer>
        <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </footer>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created index file: {index_path}")

def main():
    """Main function to run the entire analysis and report generation"""
    print("=" * 60)
    print("DBSCAN Customer Clustering Analysis Report Generator")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create the report directory
    report_dir = create_report_directory()
    
    # Run the visualization script
    vis_dir = run_visualization_script()
    
    # Copy the reports and visualizations
    copy_markdown_report(report_dir)
    copy_visualizations(vis_dir, report_dir)
    copy_best_cluster_stats(report_dir)
    
    # Create HTML report and index
    create_html_report(report_dir)
    create_report_index(report_dir)
    
    elapsed_time = time.time() - start_time
    print("\nReport generation completed!")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Report available at: {report_dir}/index.html")
    print("=" * 60)

if __name__ == "__main__":
    main() 
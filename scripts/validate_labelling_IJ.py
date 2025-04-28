#@ File (label="Labels Folder", style="directory") labels_folder
#@ File (label="Sources Folder", style="directory") sources_folder

from ij import IJ, WindowManager
import os
from ij import IJ, ImagePlus
from ij.process import LUT
import jarray
from ij.gui import WaitForUserDialog

def apply_label_to_RGB(imp):
    """
    Applies a custom LUT to the given ImagePlus object for label visualization.
    0=BG, 1=red, 2=green, 3=clear_blue, 4=yellow, 5=pink, 6=orange, 7=white, 8=teal
    """
    n = 9
    reds = jarray.array([0, 255,   0,   0, 255, 255, 255, 255,   0], 'B')
    greens = jarray.array([0,   0, 255, 191, 255, 105, 165, 255, 128], 'B')
    blues = jarray.array([0,   0,   0, 255,   0, 180,   0, 255, 128], 'B')
    lut = LUT(n, n, reds, greens, blues)
    imp.setLut(lut)

def get_sorted_filenames(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.tif', '.tiff'))]
    files.sort()
    return files

labels_files = get_sorted_filenames(labels_folder.getAbsolutePath())
sources_files = get_sorted_filenames(sources_folder.getAbsolutePath())

if labels_files != sources_files:
    IJ.showMessage("Error", "Label and source file names do not match or are not in the same order.")
    raise Exception("File name mismatch")

for i, (label_name, source_name) in enumerate(zip(labels_files, sources_files)):
    # Open label image
    label_path = os.path.join(labels_folder.getAbsolutePath(), label_name)
    imp_label = IJ.openImage(label_path)
    imp_label.setTitle(label_name + "-labels")
    # Apply label_to_RGB LUT
    apply_label_to_RGB(imp_label)

    # Convert to RGB
    IJ.run(imp_label, "RGB Color", "")
    imp_label.show()
    
    # Open source image
    source_path = os.path.join(sources_folder.getAbsolutePath(), source_name)
    imp_source = IJ.openImage(source_path)
    imp_source.setTitle(source_name)
    imp_source.show()

    # Make sure the source image is the active window
    WindowManager.setCurrentWindow(imp_source.getWindow())

    # Add overlay (imp_label as mask over imp_source)
    IJ.run("Add Image...", "image=" + imp_label.getTitle() + " x=0 y=0 opacity=40")

    # Wait for user to continue or cancel
    d = WaitForUserDialog("Next Image", "Click OK to continue to the next image.\nClick Cancel or press ESC to stop.")
    d.show()
    if d.escPressed():
        # User pressed Cancel or ESC, break the loop
        break

    # Close images before next iteration, without asking to save
    imp_label.changes = False
    imp_source.changes = False
    imp_label.close()
    imp_source.close()
    overlay = WindowManager.getImage("Result of " + imp_source.getTitle())
    if overlay:
        overlay.changes = False
        overlay.close()
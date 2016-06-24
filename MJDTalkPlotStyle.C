{
  //WARNING: If histograms not made with this style 
  //You will need gROOT->ForceStyle() in your macro.

  // Title, Stats, Date off by default
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptDate(0);
  
  // Axis titles
  Int_t kAxisTitleFont = 133; // 13 (Times New Roman) + 3 (size in pixels)
  Float_t kAxisTitleSize = 30;
  Float_t kAxisTitleOffset = 1.0;
  gStyle->SetTitleSize(kAxisTitleSize, "XYZ");
  gStyle->SetTitleFont(kAxisTitleFont, "XYZ");
  gStyle->SetTitleXOffset(kAxisTitleOffset);
  gStyle->SetTitleYOffset(kAxisTitleOffset);

  // Axis labels
  Int_t kLabelFont = 133;
  Float_t kLabelSize = 25;
  Float_t kLabelOffset = 0.006;
  gStyle->SetLabelFont(kLabelFont, "XYZ");
  gStyle->SetLabelSize(kLabelSize, "XYZ");
  gStyle->SetLabelOffset(kLabelOffset, "XY");
  gStyle->SetLabelOffset(kLabelOffset*0.5, "Z");

  // Other text (e.g. legends, stats)
  Int_t kTextFont = 133;
  Float_t kTextSize = 25;
  gStyle->SetTextSize(kTextSize);
  gStyle->SetTextFont(kTextFont);
  gStyle->SetTextColor(1);

  // Fill solid by default
  Int_t kFillStyle=1001;
  
  // No little lines at the ends of error bars.
  gStyle->SetEndErrorSize(0);
  
  // Canvas width and height: 600x800
  gStyle->SetCanvasDefH(600);
  gStyle->SetCanvasDefW(800);
  gStyle->SetCanvasBorderMode(0);
  
  // Pads and margins: use same margin on top / bottom and left / right for easy
  // alignment. User can crop if desired.
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadTopMargin(0.12);
  gStyle->SetPadBottomMargin(0.12);
  gStyle->SetPadRightMargin(0.12);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadTickX(0);
  gStyle->SetPadTickY(0);

  //Frame
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameLineWidth(2);
  
  // Legend: borderless with white background.
  gStyle->SetLegendBorderSize(0);
  gStyle->SetLegendFillColor(0);
  
  // Colors: use rainbow scale
  gStyle->SetPalette(1);
}


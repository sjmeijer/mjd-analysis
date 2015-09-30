//Plot the E for a given set of runs, for a given channel, with a given energy parameter

#include "TFile.h"
#include "TTree.h"
 #include "TCanvas.h"
 #include "TCut.h"
 #include "TH1.h"
#include "TAxis.h"
#include "TPaveText.h"


 void plot_energy(){
     
     TCut cut_chan = "channel==674 && trapECal > 1700";
     std::string energyParam = "trapECal";
     
    //-------------Data Info
     std::string prefix = "/Users/bshanks/Dev/benGatData/surfmjd/data/gatified/P3JDY/";
     //std::string prefix = "/project/projectdirs/majorana/data/mjd/surfprot/data/gatified/P3CK3/";
     int startrun = 6964;
     int endrun = 6964;
     
     int pulserLo = 135*pow(10,3);
     int pulserHi = 138*pow(10,3);
     
     
     //std::string prefix = "~/Desktop/a_e_ch152/";
     //int startrun = 40003505;
     //int endrun = 40003505;
     
     std::string treename = "mjdTree";
     std::string infilename = "mjd_run";
     //Make sure you change the channel you want to run on
     
     
     
     //This can also handle more than 1 run
     TChain *t = new TChain(treename.c_str());
     
     for(int i=startrun;i<=endrun;i++){
         char number[1000];
         sprintf(number,"%d",i);
         //printf("The number is somehow %d \n",i);
         std::string fullfilename = prefix+infilename+number+".root";
         t->AddFile(fullfilename.c_str());
         cout<<"Now Loading "<<fullfilename.c_str()<<endl;
     }

     
     // The canvas on which we'll draw the graph
     TCanvas* c1 = new TCanvas();
     
     c1->cd(1);
     //c1->SetLogy();
     c1->SetMargin(0.15, 0.1, 0.15, 0.1);
     
     t->Draw(energyParam.c_str(), cut_chan);
//     TH1F *htemp = (TH1F*)gPad->GetPrimitive("htemp");
//     htemp->GetEntries();
//     
//     htemp->SetStats(0);
//     htemp->GetXaxis()->SetTitle("Energy (uncal)");
//     htemp->GetXaxis()->SetTitleSize(0.06);
//     htemp->GetYaxis()->SetTitle("Counts");
//     htemp->GetYaxis()->CenterTitle();
//     htemp->GetYaxis()->SetTitleSize(0.06);
     
     c1->Update();
     
     c1->Print("E_spectrum.pdf");
     c1->Print("E_spectrum.root");
     
 }

 #ifndef __CINT__
 int main(){
     psa_double_peaking_plots();
     }
 #endif

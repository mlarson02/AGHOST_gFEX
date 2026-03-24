// Run with: root -b -l -q 'gFEX_EB_Ntupler.C'
#include <algorithm>
#include <numeric>   
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <filesystem> 
#include <algorithm>  
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include <unordered_map>
#include <stdexcept>

#include "TXMLEngine.h"

// Ntupler helper functions

struct WeightInfo {
    double weight;
    bool unbiased;
};

std::unordered_map<unsigned int, WeightInfo> eventWeightMap;

void loadWeights(const std::string& xmlFileName) {

    TXMLEngine xml;

    XMLDocPointer_t xmldoc = xml.ParseFile(xmlFileName.c_str());
    if (!xmldoc) {
        std::cerr << "ERROR: Cannot parse XML file\n";
        return;
    }

    XMLNodePointer_t root = xml.DocGetRootElement(xmldoc); // <run>

    XMLNodePointer_t child = xml.GetChild(root);

    std::unordered_map<int, WeightInfo> weightIdMap;

    // -----------------------------
    // Loop over <weights> and <events>
    // -----------------------------
    for (; child != nullptr; child = xml.GetNext(child)) {

        const char* nodeName = xml.GetNodeName(child);

        // -----------------------------
        // Parse <weights>
        // -----------------------------
        if (strcmp(nodeName, "weights") == 0) {

            XMLNodePointer_t wnode = xml.GetChild(child);

            for (; wnode != nullptr; wnode = xml.GetNext(wnode)) {

                int id = atoi(xml.GetAttr(wnode, "id"));
                double value = atof(xml.GetAttr(wnode, "value"));
                int unbiasedInt = atoi(xml.GetAttr(wnode, "unbiased"));

                weightIdMap[id] = {value, (bool)unbiasedInt};
            }
        }

        // -----------------------------
        // Parse <events>
        // -----------------------------
        if (strcmp(nodeName, "events") == 0) {

            XMLNodePointer_t enode = xml.GetChild(child);

            for (; enode != nullptr; enode = xml.GetNext(enode)) {

                unsigned int eventNumber = atoi(xml.GetAttr(enode, "n"));
                int weightId = atoi(xml.GetAttr(enode, "w"));

                if (weightIdMap.find(weightId) != weightIdMap.end()) {
                    eventWeightMap[eventNumber] = weightIdMap[weightId];
                } else {
                    std::cerr << "WARNING: Unknown weight id " << weightId << "\n";
                }
            }
        }
    }

    std::cout << "Loaded " << eventWeightMap.size() << " event weights\n";

    xml.FreeDoc(xmldoc);
}

// Main function
void nTupler() {

    TString outputFileName = "gFEX_EB_Ntuple.root";
    
    // Create ROOT output file
    TFile* outputFile = new TFile(outputFileName, "RECREATE");

    TTree* eventInfoTree = new TTree("eventInfoTree", "Tree storing event information (e.g., mcEventWeight) required for rate computation");
    TTree* gFexSRJTree = new TTree("gFexSRJTree", "Tree storing event-wise Et, Eta, Phi");
    TTree* gFexLeadingSRJTree = new TTree("gFexLeadingSRJTree", "Tree storing event-wise Et, Eta, Phi");
    TTree* gFexSubleadingSRJTree = new TTree("gFexSubleadingSRJTree", "Tree storing event-wise Et, Eta, Phi");
    TTree* gFexLRJTree = new TTree("gFexLRJTree", "Tree storing event-wise Et, Eta, Phi");
    TTree* gFexLeadingLRJTree = new TTree("gFexLeadingLRJTree", "Tree storing event-wise Et, Eta, Phi");
    TTree* gFexSubleadingLRJTree = new TTree("gFexSubleadingLRJTree", "Tree storing event-wise Et, Eta, Phi");
    TTree* gFEXMHTJwoJTree = new TTree("gFEXMHTJwoJTree", "Tree storing event-wise Et, Eta, Phi");
    TTree* gFEXMSTJwoJTree = new TTree("gFEXMSTJwoJTree", "Tree storing event-wise Et, Eta, Phi");
    TTree* gFEXMETJwoJTree = new TTree("gFEXMETJwoJTree", "Tree storing event-wise Et, Eta, Phi");
    TTree* gFEXScalarMETJwoJTree = new TTree("gFEXScalarMETJwoJTree", "Tree storing event-wise Et, Eta, Phi");


    double eventWeight;
    bool eventBiasedFlag; 

    // L1Calo jets vectors
    std::vector<double> gFexSRJEtValues, gFexSRJEtaValues, gFexSRJPhiValues;
    std::vector<double> gFexLRJEtValues, gFexLRJEtaValues, gFexLRJPhiValues;
    //std::vector<double> gFexSRJLeadingEtValues, gFexSRJLeadingEtaValues, gFexSRJLeadingPhiValues;
    //std::vector<double> gFexSRJSubleadingEtValues, gFexSRJSubleadingEtaValues, gFexSRJSubleadingPhiValues;
    //std::vector<double> gFexLRJLeadingEtValues, gFexLRJLeadingEtaValues, gFexLRJLeadingPhiValues;
    //std::vector<double> gFexLRJSubleadingEtValues, gFexLRJSubleadingEtaValues, gFexLRJSubleadingPhiValues;

    double mhx, mhy;
    int mhxDigi, mhyDigi;
    double msx, msy;
    int msxDigi, msyDigi;
    double metx, mety;
    int metxDigi, metyDigi;
    double met, sumEt; 
    unsigned int metDigi, sumEtDigi;


    eventInfoTree->Branch("eventWeight", &eventWeight);
    eventInfoTree->Branch("eventBiasedFlag", &eventBiasedFlag);

    // gFexSRJTree
    gFexSRJTree->Branch("Et", &gFexSRJEtValues);
    gFexSRJTree->Branch("Eta", &gFexSRJEtaValues);
    gFexSRJTree->Branch("Phi", &gFexSRJPhiValues);

    // gFexLeadingSRJTree
    /*gFexLeadingSRJTree->Branch("Et", &gFexSRJLeadingEtValues);
    gFexLeadingSRJTree->Branch("Eta", &gFexSRJLeadingEtaValues);
    gFexLeadingSRJTree->Branch("Phi", &gFexSRJLeadingPhiValues);

    // gFexSubleadingSRJTree
    gFexSubleadingSRJTree->Branch("Et", &gFexSRJSubleadingEtValues);
    gFexSubleadingSRJTree->Branch("Eta", &gFexSRJSubleadingEtaValues);
    gFexSubleadingSRJTree->Branch("Phi", &gFexSRJSubleadingPhiValues);*/

    // gFexLRJTree
    gFexLRJTree->Branch("Et", &gFexLRJEtValues);
    gFexLRJTree->Branch("Eta", &gFexLRJEtaValues);
    gFexLRJTree->Branch("Phi", &gFexLRJPhiValues);

    // gFexLeadingLRJTree
    /*gFexLeadingLRJTree->Branch("Et", &gFexLRJLeadingEtValues);
    gFexLeadingLRJTree->Branch("Eta", &gFexLRJLeadingEtaValues);
    gFexLeadingLRJTree->Branch("Phi", &gFexLRJLeadingPhiValues);

    // gFexSubleadingLRJTree
    gFexSubleadingLRJTree->Branch("Et", &gFexLRJSubleadingEtValues);
    gFexSubleadingLRJTree->Branch("Eta", &gFexLRJSubleadingEtaValues);
    gFexSubleadingLRJTree->Branch("Phi", &gFexLRJSubleadingPhiValues);*/

    gFEXMHTJwoJTree->Branch("mhx", &mhx);
    gFEXMHTJwoJTree->Branch("mhy", &mhy);
    gFEXMHTJwoJTree->Branch("mhxDigi", &mhxDigi);
    gFEXMHTJwoJTree->Branch("mhyDigi", &mhyDigi);

    gFEXMSTJwoJTree->Branch("msx", &msx);
    gFEXMSTJwoJTree->Branch("msy", &msy);
    gFEXMSTJwoJTree->Branch("msxDigi", &msxDigi);
    gFEXMSTJwoJTree->Branch("msyDigi", &msyDigi);

    gFEXMETJwoJTree->Branch("metx", &metx);
    gFEXMETJwoJTree->Branch("mety", &mety);
    gFEXMETJwoJTree->Branch("metxDigi", &metxDigi);
    gFEXMETJwoJTree->Branch("metyDigi", &metyDigi);
    
    gFEXScalarMETJwoJTree->Branch("met", &met);
    gFEXScalarMETJwoJTree->Branch("sumEt", &sumEt);
    gFEXScalarMETJwoJTree->Branch("metDigi", &metDigi);
    gFEXScalarMETJwoJTree->Branch("sumEtDigi", &sumEtDigi);

    // Main processing loop
    std::string fileName = "/data/larsonma/AGHOST_gFEX/AOD_physics_EnhancedBias_run00500306_lb-all_SFO-all.root";
    std::cout << "Processing file: " << fileName << endl;

    std::string weightFileName = "/data/larsonma/AGHOST_gFEX/EnhancedBiasWeights_500306.xml";
    loadWeights(weightFileName);

    TFile* f = TFile::Open(fileName.c_str());
    if (!f || f->IsZombie()) {
        cerr << "Could not open " << fileName << endl;
    }

    // Open the EB file
    xAOD::TEvent event(xAOD::TEvent::kClassAccess);
    if (!event.readFrom(f).isSuccess()) {
        cerr << "Cannot read xAOD from file." << endl;
    }

    std::cout << "  Number of events: " << event.getEntries() << endl;
    //for (Long64_t iEvt = 0; iEvt < 100; ++iEvt) {
    for (Long64_t iEvt = 0; iEvt < event.getEntries(); ++iEvt) {
        //std::cout << "-----------------------" << "\n";
        //std::cout << "iEvt: " << iEvt << "\n";
        // Retrieve truth and in time pileup jets first to apply hard-scatter-softer-than-PU (HSTP) filter (described here: https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/JetEtMissMCSamples#Dijet_normalization_procedure_HS)
        // Also require that truth jet collection is not empty
        event.getEntry(iEvt);               // DAOD event iEvt

        if ((iEvt % 10000) == 0) std::cout << "iEvt: " << iEvt << "\n";
        
        // -- retrieve collections from DAOD ---
        const xAOD::EventInfo_v1* EventInfo = nullptr;
        if (!event.retrieve(EventInfo, "EventInfo").isSuccess()) {
            cerr << "Cannot access EventInfo" << endl;
            continue;
        }

        const xAOD::gFexJetRoIContainer* L1_gFexSRJetRoI = nullptr;
        if (!event.retrieve(L1_gFexSRJetRoI, "L1_gFexSRJetRoI").isSuccess()) {
            std::cerr << "Failed to retrieve gFex SR jets" << std::endl;
        }

        const xAOD::gFexJetRoIContainer* L1_gFexLRJetRoI = nullptr;
        if (!event.retrieve(L1_gFexLRJetRoI, "L1_gFexLRJetRoI").isSuccess()) {
            cerr << "Failed to retrieve gFex LR jets" << endl;
        }
/*
        /// Methods that require combining results or applying scales
0134 
0135    /// Methods to convert the TOB's quantities accordingto the resolution provided
0136    /// METquantityOne() is for converting all quantityOne of MET TOBs (METx, MHTx, MSTx, MET)
0137    float gFexGlobalRoI_v1::METquantityOne() const {
0138     if (globalType() != gNull){
0139         return quantityOne()*tobEtScaleOne();
0140     }
0141     return -999;
0142    }
0143    /// METquantityTwo() is for converting all quantityTwo of MET TOBs, except SumEt (METy, MHTy, MSTy)
0144    /// Note that the scale to be used is still ScaleOne
0145    float gFexGlobalRoI_v1::METquantityTwo() const {
0146     if (globalType() != gNull){
0147       return quantityTwo()*tobEtScaleOne();
0148     }
0149 
0150     return -999;
0151    }
0152    /// SumEt() is for converting SumEt (METy, MHTy, MSTy)
0153    /// Note that SumEt is currently using scale one (200 MeV), but might change in the future
0154    float gFexGlobalRoI_v1::SumEt() const {
0155     if (globalType() == gScalar ){
0156         return quantityTwo()*tobEtScaleOne();
0157     }
0158     return -999;
0159    }
0160 */
        const DataVector<xAOD::gFexGlobalRoI_v1>* L1_gMHTComponentsJwoj = nullptr;
        if (!event.retrieve(L1_gMHTComponentsJwoj, "L1_gMHTComponentsJwoj").isSuccess()) {
            std::cerr << "Failed to retrieve MET soft term" << std::endl;
        }

        const DataVector<xAOD::gFexGlobalRoI_v1>* L1_gMETComponentsJwoj = nullptr;
        if (!event.retrieve(L1_gMETComponentsJwoj, "L1_gMETComponentsJwoj").isSuccess()) {
            std::cerr << "Failed to retrieve MET" << std::endl;
        }

        const DataVector<xAOD::gFexGlobalRoI_v1>* L1_gMSTComponentsJwoj = nullptr;
        if (!event.retrieve(L1_gMSTComponentsJwoj, "L1_gMSTComponentsJwoj").isSuccess()) {
            std::cerr << "Failed to retrieve MET soft term" << std::endl;
        }

        const DataVector<xAOD::gFexGlobalRoI_v1>* L1_gScalarEJwoj = nullptr;
        if (!event.retrieve(L1_gScalarEJwoj, "L1_gScalarEJwoj").isSuccess()) {
            std::cerr << "Failed to retrieve MET soft term" << std::endl;
        }
        
        const xAOD::gFexJetRoIContainer* L1_gFexRhoRoI = nullptr;
        if (!event.retrieve(L1_gFexRhoRoI, "L1_gFexRhoRoI").isSuccess()) {
            std::cerr << "Failed to retrieve gFex energy density" << std::endl;
        }

        unsigned int eventNumber = EventInfo->eventNumber();

        auto it = eventWeightMap.find(eventNumber);
        if (it != eventWeightMap.end()) {
            eventWeight = it->second.weight;
            eventBiasedFlag = it->second.unbiased;
        } else {
            std::cerr << "WARNING: No weight for event " << eventNumber << "\n";
        }

        // Now use them
        //std::cout << "eventNumber = " << eventNumber
        //        << ", weight = " << eventWeight
        //        << ", unbiased = " << eventBiasedFlag << std::endl;

        gFexLRJEtValues.clear();
        gFexLRJEtaValues.clear();
        gFexLRJPhiValues.clear();

        gFexSRJEtValues.clear();
        gFexSRJEtaValues.clear();
        gFexSRJPhiValues.clear();

        // Fill vectors in sorted order
        for (size_t i = 0; i < L1_gFexSRJetRoI->size(); ++i) {
            const auto& jet = (*L1_gFexSRJetRoI)[i];
            double et = jet->et() / 1000.0; // FIXME update units based on Paula's script
            gFexSRJEtValues.push_back(et);
            gFexSRJEtaValues.push_back(jet->eta());
            gFexSRJPhiValues.push_back(jet->phi());
        }

        // Fill vectors in unsorted order
        for (size_t i = 0; i < L1_gFexLRJetRoI->size(); ++i) {
            const auto& jet = (*L1_gFexLRJetRoI)[i];
            double et = jet->et() / 1000.0; // already in GeV

            gFexLRJEtValues.push_back(et);
            gFexLRJEtaValues.push_back(jet->eta());
            gFexLRJPhiValues.push_back(jet->phi());
        }
        
        //std::cout << "MET HARD TERM: " << "\n";
        for (size_t i = 0; i < L1_gMHTComponentsJwoj->size(); ++i) {
            const auto& mht = (*L1_gMHTComponentsJwoj)[i];
            mhx = mht->METquantityOne() / 1000.0; // convert to GeV
            mhxDigi = (int)mht->METquantityOne() / 200.0; // lsb = 200 MeV --> convert from MeV to digitized version
            //std::cout << "mhx: " << mhx << "\n";
            //std::cout << "mhxDigi: " << mhxDigi << "\n";
            mhy= mht->METquantityTwo() / 1000.0; // convert to GeV
            mhyDigi = (int)mht->METquantityTwo() / 200.0; // lsb = 200 MeV --> convert from MeV to digitized version
            //std::cout << "mhy: " << mhy << "\n";
            //std::cout << "mhyDigi: " << mhyDigi << "\n";
        }
        
        //std::cout << "MET SOFT TERM: " << "\n";
        for (size_t i = 0; i < L1_gMSTComponentsJwoj->size(); ++i) {
            const auto& mst = (*L1_gMSTComponentsJwoj)[i];
            msx = mst->METquantityOne() / 1000.0; // Convert to GeV
            msxDigi = (int)mst->METquantityOne() / 200.0; // lsb = 200 MeV --> convert from MeV to digitized version
            //std::cout << "msx: " << msx << "\n";
            msy = mst->METquantityTwo() / 1000.0; // Convert to GeV
            msyDigi = (int)mst->METquantityTwo() / 200.0; // lsb = 200 MeV --> convert from MeV to digitized version
            //std::cout << "msy: " << msy << "\n";
        }

        //std::cout << "MET HARD + SOFT TERM?: " << "\n";
        for (size_t i = 0; i < L1_gMETComponentsJwoj->size(); ++i) {
            const auto& metTotal = (*L1_gMETComponentsJwoj)[i];
            metx = metTotal->METquantityOne() / 1000.0; // Convert to GeV
            metxDigi = (int)metTotal->METquantityOne() / 200.0; // lsb = 200 MeV --> convert from MeV to digitized version
            //std::cout << "metx: " << metx << "\n";
            mety = metTotal->METquantityTwo() / 1000.0;
            metyDigi = (int)metTotal->METquantityTwo() / 200.0; // lsb = 200 MeV --> convert from MeV to digitized version
            //std::cout << "mety: " << mety << "\n";
        }

        //std::cout << "MET SCALARS: " << "\n";
        for (size_t i = 0; i < L1_gScalarEJwoj->size(); ++i) {
            const auto& metScalar = (*L1_gScalarEJwoj)[i];
            met = metScalar->METquantityOne() / 1000.0; // Convert to GeV
            metDigi = (unsigned int)metScalar->METquantityOne() / 200.0; // lsb = 200 MeV --> convert from MeV to digitized version
            //std::cout << "met: " << met << "\n";
            //std::cout << "metDigi: " << metDigi << "\n";
            //auto met2 = metScalar->METquantityTwo();
            //std::cout << "met2: " << met2 << "\n";
            sumEt = metScalar->SumEt() / 1000.0; // Convert to GeV
            sumEtDigi = (unsigned int)metScalar->SumEt() / 200.0; // lsb = 200 MeV --> convert from MeV to digitized version
            //std::cout << "sumEt: " << sumEt << "\n";
            //std::cout << "sumEtDigi: " << sumEtDigi << "\n";
        }

        for (size_t i = 0; i < L1_gFexRhoRoI->size(); ++i) {
            const auto& rho = (*L1_gMETComponentsJwoj)[i];
        }

        eventInfoTree->Fill();
        gFexSRJTree->Fill();
        gFexLRJTree->Fill();
        //gFexLeadingSRJTree->Fill();
        //gFexSubleadingSRJTree->Fill();
        //gFexLeadingLRJTree->Fill();
        //gFexSubleadingLRJTree->Fill();
        gFEXMHTJwoJTree->Fill();
        gFEXMSTJwoJTree->Fill();
        gFEXMETJwoJTree->Fill();
        gFEXScalarMETJwoJTree->Fill();

    } // loop through events
    f->Close();
    outputFile->cd();
    std::cout << "writing output file" << "\n";
    gFexSRJTree->Write("", TObject::kOverwrite);
    gFexLRJTree->Write("", TObject::kOverwrite);
    eventInfoTree->Write("", TObject::kOverwrite);
    //gFexLeadingSRJTree->Write("", TObject::kOverwrite);
    //gFexSubleadingSRJTree->Write("", TObject::kOverwrite);
    //gFexLeadingLRJTree->Write("", TObject::kOverwrite);
    //gFexSubleadingLRJTree->Write("", TObject::kOverwrite);
    gFEXMHTJwoJTree->Write("", TObject::kOverwrite);
    gFEXMSTJwoJTree->Write("", TObject::kOverwrite);
    gFEXMETJwoJTree->Write("", TObject::kOverwrite);
    gFEXScalarMETJwoJTree->Write("", TObject::kOverwrite);
    outputFile->Close();
} // ntupler function

// processes all jzSlices + signal
void gFEX_EB_Ntupler(){
    gSystem->Load("libxAODRootAccess");
    xAOD::Init().ignore();
    gSystem->RedirectOutput("NTupler.log", "w");

    nTupler(); // call for signal (hh->4b ggF)

    gSystem->Exit(0);
}

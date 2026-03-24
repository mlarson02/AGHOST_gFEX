import uproot
import pandas as pd

f = uproot.open("gFEX_EB_Ntuple.root")

df = pd.DataFrame({
    "mhxDigi": f["gFEXMHTJwoJTree"]["mhxDigi"].array(library="np"),
    "mhyDigi": f["gFEXMHTJwoJTree"]["mhyDigi"].array(library="np"),
    "msxDigi": f["gFEXMSTJwoJTree"]["msxDigi"].array(library="np"),
    "msyDigi": f["gFEXMSTJwoJTree"]["msyDigi"].array(library="np"),
    "metxDigi": f["gFEXMETJwoJTree"]["metxDigi"].array(library="np"),
    "metyDigi": f["gFEXMETJwoJTree"]["metyDigi"].array(library="np"),
    "metDigi": f["gFEXScalarMETJwoJTree"]["metDigi"].array(library="np"),
    "sumEtDigi": f["gFEXScalarMETJwoJTree"]["sumEtDigi"].array(library="np"),
    "eventWeight": f["eventInfoTree"]["eventWeight"].array(library="np"),
    "eventBiasedFlag": f["eventInfoTree"]["eventBiasedFlag"].array(library="np"),
})

df.to_parquet("gFEX_digitizedMET.parquet", index=False)
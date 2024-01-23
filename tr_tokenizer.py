# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 14:39:11 2023

@author: VuralBayraklii
"""

# gRPC üzerinden Zemberek dil işleme servislerini kullanmak için gerekli modülleri ve paketleri içe aktar
import sys
import unicodedata

import grpc
import zemberek_grpc.language_id_pb2 as z_langid
import zemberek_grpc.language_id_pb2_grpc as z_langid_g
import zemberek_grpc.normalization_pb2 as z_normalization
import zemberek_grpc.normalization_pb2_grpc as z_normalization_g
import zemberek_grpc.preprocess_pb2 as z_preprocess
import zemberek_grpc.preprocess_pb2_grpc as z_preprocess_g
import zemberek_grpc.morphology_pb2 as z_morphology
import zemberek_grpc.morphology_pb2_grpc as z_morphology_g

# gRPC kanalını belirtilen adres ve port üzerinden oluştur
channel = grpc.insecure_channel('localhost:6789')

# Dil tespiti için servis istemcisini oluştur
langid_stub = z_langid_g.LanguageIdServiceStub(channel)

# Normalizasyon için servis istemcisini oluştur
normalization_stub = z_normalization_g.NormalizationServiceStub(channel)

# Metin ön işleme için servis istemcisini oluştur
preprocess_stub = z_preprocess_g.PreprocessingServiceStub(channel)

# Morfoloji analizi için servis istemcisini oluştur
morphology_stub = z_morphology_g.MorphologyServiceStub(channel)

# Dil tespiti fonksiyonu
def find_lang_id(i):
    response = langid_stub.Detect(z_langid.LanguageIdRequest(input=i))
    return response.langId

# Metni token'lara ayıran fonksiyon
def tokenize(i):
    response = preprocess_stub.Tokenize(z_preprocess.TokenizationRequest(input=i))
    return response.tokens

# Decode işlemini gerçekleştiren fonksiyon
def fix_decode(text):
    """Pass decode."""
    if sys.version_info < (3, 0):
        print("burada")
        return text.decode('utf-8')
    else:
        return text
    
# Morfoloji analizi fonksiyonu
def analyze(i):
    response = morphology_stub.AnalyzeSentence(z_morphology.SentenceAnalysisRequest(input=i))
    return response;



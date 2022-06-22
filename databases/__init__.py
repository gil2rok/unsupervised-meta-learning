# start my code
import sys
project_root_address = '/home/gt2453_columbia_edu/mel_umtra'
sys.path.insert(0, project_root_address)
# end my code

from .maml_benchmarks import OmniglotDatabase, MiniImagenetDatabase, Omniglot84x84Database
from .face_recognition_benchmarks import CelebADatabase, LFWDatabase, VGGFace2Database
from .cdml_challenge_benchmarks import EuroSatDatabase, PlantDiseaseDatabase, ISICDatabase, ChestXRay8Database
from .meta_dataset import CUBDatabase, AirplaneDatabase, DTDDatabase, VGGFlowerDatabase, TrafficSignDatabase, \
    MSCOCODatabase, FungiDatabase
from .audio import VoxCelebDatabase

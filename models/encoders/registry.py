"""Recipe names -> frozen feature encoders; adding a variant needs one registry entry."""
from models.encoders.load import FrozenEncoders
from models.encoders.rn50 import FrozenRN50Encoders, STACKS

from utils.experiments import ENCODERS


def build_encoder(cache, *, stack, **options):
    cls = FrozenRN50Encoders if stack in STACKS else FrozenEncoders
    return cls(cache, stack=stack, **options)

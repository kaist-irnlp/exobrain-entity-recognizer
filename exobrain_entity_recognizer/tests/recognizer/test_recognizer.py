#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `exobrain_entity_recognizer` package."""
from subprocess import call

import pytest

from click.testing import CliRunner

from ...recognizer import EntityRecognizer
from ... import cli
import spacy


@pytest.fixture(scope='session')
def models():
    call(['python', '-m', 'spacy', 'download', 'en'])
    call(['python', '-m',  'textblob.download_corpora'])


@pytest.fixture(scope='module')
def recognizer(models):
    return EntityRecognizer('en')


@pytest.fixture(scope='module')
def text():
    return """
        spaCy excels at large-scale information extraction tasks. 
        It's written from the ground up in carefully memory-managed Cython. 
        Independent research has confirmed that spaCy is the fastest in the world. 
        If your application needs to process entire web dumps, spaCy is the library you want to be using.
    """


def test_entity_recognizer(text, recognizer):
    # run recognizer
    for ent in recognizer(text):
        assert ent != {}

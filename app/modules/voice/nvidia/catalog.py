"""Loads config/services.{cloud,local}.yaml and merges them.

A reachable local NIM entry shadows its cloud counterpart, so hosted-vs-self-hosted
is a config flip with no code branch."""

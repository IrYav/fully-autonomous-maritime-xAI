from collections.abc import Iterator, Sequence
from matplotlib.artist import Artist
from matplotlib.patches import PathPatch
from matplotlib.path import Path

ArtistLike = Artist | Sequence[Artist]

class ArtistManager:
  '''
    Small manager class to handle list artists,
    overwriting artists
    and consistent application of visibility checkboxes
  '''
  def __init__(self):
    self._artists: dict[str, ArtistLike] = {}

  def get(self, key, default = None):
    return self._artists.get(key,default)

  def __getitem__(self, key):
    return self._artists[key]

  def __setitem__(self, key, artist: ArtistLike):
    '''
      Set a artist in a key, maintain visibility setting,
      if it was predefined. Visible by default.
    '''
    rep = self._rep_key(key) 
    vis = rep is None or rep.get_visible()
    for a in self._iter_artist(artist):
      a.set_visible(vis)
    if key in self._artists:
      del self[key]
    self._artists[key] = artist
  
  def __delitem__(self, key):
    for a in self._iter_key(key):
      a.remove()

    self._artists.pop(key, None)

  def __contains__(self, key):
    return key in self._artists

  def __iter__(self):
    return iter(self._artists)

  def toggle_vis(self, key):
    '''
      Toggles visibility of the relevant artist.

      Also works with (not yet) initialized keys, 
      in which case a dummy object is created (and set to invisible).
    '''
    rep = self._rep_key(key)

    if rep is None:
      #Dummy object to ensure toggle comes through.
      self[key] = Artist().set_visible(False)
      return #No artists
    
    visible = rep.get_visible()

    for a in self._iter_key(key):
      a.set_visible(not visible)

  def set_vis(self, key, visible: bool):
    for a in self._iter_key(key):
      a.set_visible(visible)

  # --- Properties ---
  DEFAULT_INVIS = {
    "mesh": False,
    "gradient": False, # Quiver
    "potential_fill": False,
    "potential_contour": False,
    "corridor": False,
    "last_step": False
  }

  CHECKBOX_ARTISTS = {
    "islands": None,
    "end" : None,
    "ship" : None,
    "last_step" : None,
    "total_trajectory" : None,
    "corridor": None,
    "mesh": None,
    "potential_contour": None,
    "potential_fill": None,
    "gradient": None,
  }

  # --- Auxiliaries ---

  def _iter_key(self, key) -> Iterator[Artist]:
    return self._iter_artist(self.get(key))

  def _rep_key(self, key) -> Artist | None:
    '''Gets a representative from a key (if any)'''
    return self._rep_artist(self.get(key))

  @staticmethod
  def _iter_artist(artist: ArtistLike | None) -> Iterator[Artist]:
    if artist is None:
      return
    if isinstance(artist, Artist):
      yield artist
    else:
      yield from artist

  @staticmethod
  def _rep_artist(artist: ArtistLike | None) -> Artist | None:
    '''Gets a representative (if any)'''
    return next(ArtistManager._iter_artist(artist), None)
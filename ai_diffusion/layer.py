from __future__ import annotations
from contextlib import nullcontext
from enum import Enum
import krita
from PyQt5.QtCore import QObject, QUuid, QByteArray, QTimer, pyqtSignal
from PyQt5.QtGui import QImage

from .image import Extent, Bounds, Image
from .util import ensure
from . import eventloop


class LayerType(Enum):
    paint = "paintlayer"
    vector = "vectorlayer"
    group = "grouplayer"
    file = "filelayer"
    clone = "clonelayer"
    filter = "filterlayer"
    transparency = "transparencymask"
    selection = "selectionmask"

    @property
    def is_image(self):
        return not self.is_mask

    @property
    def is_mask(self):
        return self in [LayerType.transparency, LayerType.selection]


class Layer(QObject):
    """Wrapper around a Krita Node. Provides pythonic interface, read and write pixels
    from/to QImage. Exposes some events based on polling done in LayerManager.
    Layer objects are cached, there is a guarantee only one instance exists per layer node.
    """

    _observer: LayerManager
    _node: krita.Node
    _name: str

    renamed = pyqtSignal(str)
    removed = pyqtSignal()

    def __init__(self, observer: LayerManager, node: krita.Node):
        super().__init__()
        self._observer = observer
        self._node = node
        self._name = node.name()

    @property
    def id(self):
        return self._node.uniqueId()

    @property
    def id_string(self):
        return self._node.uniqueId().toString()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if self._name == value:
            return
        self._name = value
        self._node.setName(value)
        self.renamed.emit(value)

    @property
    def type(self):
        return LayerType(self._node.type())

    @property
    def was_removed(self):
        return self._observer.updated().find(self.id) is None

    @property
    def is_visible(self):
        return self._node.visible()

    @is_visible.setter
    def is_visible(self, value):
        self._node.setVisible(value)

    def hide(self):
        self._node.setVisible(False)
        self.refresh()

    def show(self):
        self._node.setVisible(True)
        self.refresh()

    @property
    def is_active(self):
        return self is self._observer.active

    @property
    def is_locked(self):
        return self._node.locked()

    @is_locked.setter
    def is_locked(self, value):
        self._node.setLocked(value)

    @property
    def bounds(self):
        return Bounds.from_qrect(self._node.bounds())

    @property
    def parent_layer(self):
        if parent := self._node.parentNode():
            return self._observer.wrap(parent)
        return None

    @property
    def child_layers(self):
        return [self._observer.wrap(child) for child in self._node.childNodes()]

    @property
    def is_root(self):
        return self._node.parentNode() is None

    def get_pixels(self, bounds: Bounds | None = None, time: int | None = None):
        bounds = bounds or self.bounds
        if time is None:
            data: QByteArray = self._node.projectionPixelData(*bounds)
        else:
            data: QByteArray = self._node.pixelDataAtTime(time, *bounds)
        assert data is not None and data.size() >= bounds.extent.pixel_count * 4
        return Image(QImage(data, *bounds.extent, QImage.Format.Format_ARGB32))

    def write_pixels(self, img: Image, bounds: Bounds | None = None, make_visible=True):
        layer_bounds = self.bounds
        bounds = bounds or layer_bounds
        if layer_bounds != bounds and not layer_bounds.is_zero:
            # layer.cropNode(*bounds)  <- more efficient, but clutters the undo stack
            blank = Image.create(layer_bounds.extent, fill=0)
            self._node.setPixelData(blank.data, *layer_bounds)
        self._node.setPixelData(img.data, *bounds)
        if make_visible:
            self.show()
        if self.is_visible:
            self.refresh()

    def get_mask(self, bounds: Bounds | None):
        bounds = bounds or self.bounds
        if self.type.is_mask:
            data: QByteArray = self._node.pixelData(*bounds)
            assert data is not None and data.size() >= bounds.extent.pixel_count
            return Image(QImage(data, *bounds.extent, QImage.Format.Format_Grayscale8))
        else:
            img = self.get_pixels(bounds)
            alpha = img._qimage.convertToFormat(QImage.Format.Format_Alpha8)
            alpha.reinterpretAsFormat(QImage.Format.Format_Grayscale8)
            return Image(alpha)

    def move_to_top(self):
        parent = self._node.parentNode()
        if parent.childNodes()[-1] == self._node:
            return  # already top-most layer
        with RestoreActiveLayer(self._observer):
            parent.removeChildNode(self.node)
            parent.addChildNode(self.node, None)

    def refresh(self):
        # Hacky way of refreshing the projection of a layer, avoids a full document refresh
        self._node.setBlendingMode(self._node.blendingMode())

    def thumbnail(self, size: Extent):
        return self.node.thumbnail(*size)

    def remove(self):
        self._node.remove()
        self._observer.update()

    def compute_bounds(self):
        bounds = self.bounds
        if self.type.is_mask:
            # Unfortunately node.bounds() returns the whole image
            # Use a selection to get just the bounds that contain pixels > 0
            s = krita.Selection()
            data = self.node.pixelData(*bounds)
            s.setPixelData(data, *bounds)
            return Bounds(s.x(), s.y(), s.width(), s.height())
        elif self.type is LayerType.group:
            for child in self.child_layers:
                if child.type is LayerType.transparency:
                    bounds = child.compute_bounds()
        return bounds

    @property
    def siblings(self):
        below: list[Layer] = []
        above: list[Layer] = []
        parent = self.parent_layer

        if parent is None:
            return below, above

        current = below
        for l in parent.child_layers:
            if l == self:
                current = above
            else:
                current.append(l)
        return below, above

    @property
    def sibling_above(self):
        nodes = ensure(self.parent_layer).child_layers
        index = nodes.index(self)
        if index >= 1:
            return nodes[index - 1]
        return self

    @property
    def is_animated(self):
        return self._node.animated()

    @property
    def node(self):
        return self._node

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, Layer):
            return self.id == other.id
        return False


class RestoreActiveLayer:
    layer: Layer | None = None

    def __init__(self, layers: LayerManager):
        self._observer = layers

    def __enter__(self):
        self.layer = self._observer.active

    def __exit__(self, exc_type, exc_value, traceback):
        # Some operations like inserting a new layer change the active layer as a side effect.
        # It doesn't happen directly, so changing it back in the same call doesn't work.
        eventloop.run(self._restore())

    async def _restore(self):
        if self.layer:
            if self.layer.is_active:
                # Maybe whatever event we expected to change the active layer hasn't happened yet.
                await eventloop.wait_until(
                    lambda: self.layer is not None and not self.layer.is_active, no_error=True
                )
            self._observer.active = self.layer


class LayerManager(QObject):
    """Periodically checks the document for changes in the layer structure. Krita doesn't expose
    Python events for these kinds of changes, so we have to poll and compare.
    Provides helpers to quickly create new layers and groups with initial content.
    """

    changed = pyqtSignal()
    active_changed = pyqtSignal()

    _doc: krita.Document | None
    _root: Layer | None
    _layers: dict[QUuid, Layer]
    _active: QUuid
    _timer: QTimer

    def __init__(self, doc: krita.Document | None):
        super().__init__()
        self._doc = doc
        self._layers = {}
        if doc is not None:
            root = doc.rootNode()
            self._root = Layer(self, root)
            self._layers = {self._root.id: self._root}
            self._active = doc.activeNode().uniqueId()
            self.update()
            self._timer = QTimer()
            self._timer.setInterval(500)
            self._timer.timeout.connect(self.update)
            self._timer.start()
        else:
            self._root = None
            self._active = QUuid()

    def update(self):
        if self._doc is None:
            return
        root_node = self._doc.rootNode()
        if root_node is None:
            return  # Document has been closed

        active = self._doc.activeNode()
        if active is None:
            return

        if active.uniqueId() != self._active:
            self._active = active.uniqueId()
            self.active_changed.emit()

        removals = set(self._layers.keys())
        changes = False
        for n in traverse_layers(root_node):
            id = n.uniqueId()
            if id in self._layers:
                removals.remove(id)
                layer = self._layers[id]
                if layer.name != n.name():
                    layer.name = n.name()
                    changes = True
            else:
                self._layers[id] = Layer(self, n)
                changes = True

        removals.remove(self.root.id)
        for id in removals:
            self._layers[id].removed.emit()
            del self._layers[id]

        if removals or changes:
            self.changed.emit()

    def wrap(self, node: krita.Node) -> Layer:
        layer = self.find(node.uniqueId())
        if layer is None:
            layer = self.updated()._layers[node.uniqueId()]
        return layer

    def find(self, id: QUuid) -> Layer | None:
        if self._doc is None:
            return None
        return self._layers.get(id)

    def updated(self):
        self.update()
        return self

    @property
    def root(self):
        assert self._root is not None
        return self._root

    @property
    def active(self):
        assert self._doc is not None
        layer = self.find(self._doc.activeNode().uniqueId())
        if layer is None:
            layer = self.updated()._layers[self._active]
        return layer

    @active.setter
    def active(self, layer: Layer):
        if self._doc is not None:
            self._doc.setActiveNode(layer.node)
            self.update()

    def create(
        self,
        name: str,
        img: Image | None = None,
        bounds: Bounds | None = None,
        make_active=True,
        parent: Layer | None = None,
        above: Layer | None = None,
    ):
        doc = ensure(self._doc)
        node = doc.createNode(name, "paintlayer")
        layer = self._insert(node, parent, above, make_active)
        if img and bounds:
            layer.node.setPixelData(img.data, *bounds)
            layer.refresh()
        return layer

    def _insert(
        self,
        node: krita.Node,
        parent: Layer | None = None,
        above: Layer | None = None,
        make_active=True,
    ):
        if above is not None:
            parent = parent or above.parent_layer
        parent = parent or self.root
        with RestoreActiveLayer(self) if not make_active else nullcontext():
            parent.node.addChildNode(node, above.node if above else None)
            return self.updated().wrap(node)

    def create_vector(self, name: str, svg: str):
        doc = ensure(self._doc)
        node = doc.createVectorLayer(name)
        doc.rootNode().addChildNode(node, None)
        node.addShapesFromSvg(svg)
        layer = self.updated().wrap(node)
        layer.refresh()
        return layer

    def create_mask(self, name: str, img: Image, bounds: Bounds, parent: Layer | None = None):
        assert img.is_mask
        doc = ensure(self._doc)
        node = doc.createTransparencyMask(name)
        node.setPixelData(img.data, *bounds)
        return self._insert(node, parent=parent)

    def create_group(self, name: str, above: Layer | None = None):
        doc = ensure(self._doc)
        node = doc.createGroupLayer(name)
        return self._insert(node, above)

    def create_group_for(self, layer: Layer):
        doc = ensure(self._doc)
        group = self.wrap(doc.createGroupLayer(f"{layer.name} Group"))
        parent = ensure(layer.parent_layer, "Cannot group root layer")
        parent.node.addChildNode(group.node, layer.node)
        parent.node.removeChildNode(layer.node)
        group.node.addChildNode(layer.node, None)
        return group

    _image_types = [t.value for t in LayerType if t.is_image]
    _mask_types = [t.value for t in LayerType if t.is_mask]

    @property
    def images(self):
        if self._doc is None:
            return []
        return [self.wrap(n) for n in traverse_layers(self._doc.rootNode(), self._image_types)]

    @property
    def masks(self):
        if self._doc is None:
            return []
        return [self.wrap(n) for n in traverse_layers(self._doc.rootNode(), self._mask_types)]

    def __bool__(self):
        return self._doc is not None


def traverse_layers(node: krita.Node, type_filter: list[str] | None = None):
    for child in node.childNodes():
        yield from traverse_layers(child, type_filter)
        if not type_filter or child.type() in type_filter:
            yield child
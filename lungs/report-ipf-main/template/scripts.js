var strategies = {
    "meshViewer": function (K3DInstance, obj, id) {
        var currentMesh = null,
            jsons = K3DInstance.getWorld().ObjectsListJson;

        Object.keys(jsons).forEach(function (id) {
            jsons[id].originalOpacity = jsons[id].opacity;
        });

        setInterval(function () {
            var needsRender = false;

            Object.keys(jsons).forEach(function (id) {
                if (jsons[id].name === currentMesh) {
                    if (jsons[id].opacity < 1.0) {
                        jsons[id].opacity = 1.0;
                        K3DInstance.reload(jsons[id], { opacity: jsons[id].opacity }, true);
                        needsRender = true;
                    }
                } else {
                    if (currentMesh === null && jsons[id].opacity > jsons[id].originalOpacity) {
                        jsons[id].opacity = Math.max(jsons[id].originalOpacity, jsons[id].opacity - 0.05);
                        K3DInstance.reload(jsons[id], { opacity: jsons[id].opacity }, true);
                        needsRender = true;
                    }

                    if (currentMesh === null && jsons[id].opacity < jsons[id].originalOpacity) {
                        jsons[id].opacity = Math.min(jsons[id].originalOpacity, jsons[id].opacity + 0.05);
                        K3DInstance.reload(jsons[id], { opacity: jsons[id].opacity }, true);
                        needsRender = true;
                    }

                    if (currentMesh !== null && jsons[id].opacity > 0.05) {
                        jsons[id].opacity = Math.max(0.05, jsons[id].opacity - 0.05);
                        K3DInstance.reload(jsons[id], { opacity: jsons[id].opacity }, true);
                        needsRender = true;
                    }
                }
            });

            if (needsRender) {
                K3DInstance.render();
            }
        }, 1000 / 25);

        $("#" + id + " tr").mouseenter(function (e) {
            currentMesh = $(e.currentTarget).data('name');
        });

        $("#" + id + " tr").mouseleave(function () {
            currentMesh = null;
        });
    },
    "volumeViewer": function (K3DInstance, obj, id) {
        var slices = {},
            jsons = K3DInstance.getWorld().ObjectsListJson;

        Object.keys(jsons).forEach(function (id) {
            if (typeof (jsons[id].slice_x) !== undefined) {
                jsons[id].org_slice_x = jsons[id].slice_x;
                jsons[id].org_slice_y = jsons[id].slice_y;
                jsons[id].org_slice_z = jsons[id].slice_z;
            }

            if (typeof (jsons[id].color_range) !== undefined) {
                jsons[id].org_color_range = jsons[id].color_range.slice();
            }
        });


        function refresh(selected, center) {
            var view = $("#" + id + " select[name='view']").val() || 'z',
                ids,
                promises = [],
                jsons = K3DInstance.getWorld().ObjectsListJson;

            if (selected !== '' && typeof (selected) !== 'undefined') {
                ids = selected.split(',').map(function (v) {
                    return parseInt(v, 10);
                })
            } else {
                ids = [];
            }

            if (center) {
                center = center.split(',').map(function (v) {
                    return parseInt(v, 10);
                });
            }

            //setup volume slice
            Object.keys(jsons).forEach(function (id) {
                var changes = {};

                if (jsons[id].type === 'VolumeSlice') {
                    if (center) {
                        changes.slice_x = slices.x = center[0];
                        changes.slice_y = slices.y = center[1];
                        changes.slice_z = slices.z = center[2];
                    }

                    if (view !== 'all' && view !== '3d') {
                        ['x', 'y', 'z'].forEach(function (axis) {
                            if (axis !== view) {
                                slices[axis] = slices[axis] || jsons[id]['slice_' + axis];
                                changes['slice_' + axis] = -1;
                            } else {
                                if (jsons[id]['slice_' + axis] === -1) {
                                    changes['slice_' + axis] = slices[axis];
                                }
                            }
                        });
                    } else {
                        ['x', 'y', 'z'].forEach(function (axis) {
                            if (jsons[id]['slice_' + axis] === -1) {
                                changes['slice_' + axis] = slices[axis];
                            }
                        });
                    }

                    changes.active_masks = { data: ids };

                    Object.keys(changes).forEach(function (key) {
                        jsons[id][key] = changes[key];
                    });

                    promises.push(K3DInstance.reload(jsons[id], changes, true));
                }

                if (jsons[id].type === "Mesh") {
                    var names = jsons[id].name.split('_'),
                        maskId = parseInt(names[names.length - 1], 10),
                        visible = ids.indexOf(maskId) !== -1;

                    if (names[0] === 'mesh' && view !== 'all') {
                        visible = false;
                    }

                    if (visible) {
                        if (!jsons[id].visible) {
                            jsons[id].visible = true;
                            promises.push(K3DInstance.reload(jsons[id], { visible: true }, true));
                        }
                    } else {
                        if (jsons[id].visible) {
                            jsons[id].visible = false;
                            promises.push(K3DInstance.reload(jsons[id], { visible: false }, true));
                        }
                    }
                }
            });

            Promise.all(promises).then(function () {
                if (view === 'all') {
                    K3DInstance.setCameraMode('volume_sides');
                    K3DInstance.setCamera(K3DInstance.parameters.camera);
                } else if (view === '3d') {
                    K3DInstance.setCameraMode('trackball');
                    K3DInstance.setCamera(K3DInstance.parameters.camera);
                } else {
                    K3DInstance.setCameraMode('slice_viewer');
                    K3DInstance.setSliceViewerDirection(view);
                }

                setTimeout(K3DInstance.render, 0);
            });
        }

        $("#" + id + " select[name='filling']").change(function () {
            var val = parseFloat($(this).val());

            Object.keys(jsons).forEach(function (id) {
                if (jsons[id].type === 'VolumeSlice') {
                    jsons[id].mask_opacity = val;

                    K3DInstance.reload(jsons[id], { mask_opacity: jsons[id].mask_opacity });
                }
            });
        });

        $("#" + id + " select[name='masks']").change(function () {
            var ids = $(this).val(),
                center = $(this).find(':selected').data('center') || null;

            refresh(ids, center);
        });

        $("#" + id + " .masks_explorer tr").mouseenter(function (e) {
            refresh($(e.currentTarget).data('mask').toString());
        });

        $("#" + id + " .masks_explorer").mouseleave(function () {
            refresh($("#" + id + " select[name='masks']").val());
        });

        $("#" + id + " select[name='view']").change(function () {
            refresh(
                $("#" + id + " select[name='masks']").val()
            );
        });

        refresh(
            $("#" + id + " select[name='masks']").val(),
            $("#" + id + " select[name='masks']").find(':selected').data('center')
        );

        setTimeout(K3DInstance.render, 0);
    }
};

function initWidget(lib, obj, id, strategy) {
    var K3DInstance;

    try {
        var data = obj.data;

        if (typeof (data) === 'string') {
            data = _base64ToArrayBuffer(data);
        }

        K3DInstance = new lib.CreateK3DAndLoadBinarySnapshot(
            data,
            $("#" + id + ' .k3d').get(0)
        );

        return K3DInstance.then((K3DInstance) => {
            strategies[strategy](K3DInstance, obj, id);

            setTimeout(() => {
                obj.camera = K3DInstance.getWorld().controls.getCameraArray();
            }, 1000);

            return K3DInstance;
        });
    } catch (e) {
        console.log(e);
        return;
    }
}

$(".reset_view").click(function () {
    var wid = $(this).parents('section').attr('id');

    widgets[wid].then(function (widget) {
        var jsons = widget.getWorld().ObjectsListJson;

        widget.setCamera(widgetsData[wid].camera);

        Object.keys(jsons).forEach(function (id) {
            var change = {}

            if (typeof (jsons[id].slice_x) !== 'undefined') {
                jsons[id].slice_x = jsons[id].org_slice_x;
                jsons[id].slice_y = jsons[id].org_slice_y;
                jsons[id].slice_z = jsons[id].org_slice_z;

                change.slice_x = jsons[id].slice_x;
                change.slice_y = jsons[id].slice_y;
                change.slice_z = jsons[id].slice_z;
            }

            if (typeof (jsons[id].color_range) !== 'undefined' &&
                jsons[id].color_range.length !== 0 &&
                jsons[id].org_color_range.length !== 0) {

                jsons[id].color_range = jsons[id].org_color_range.slice();
                change.color_range = jsons[id].color_range;
            }

            if (Object.keys(change).length > 0) {
                widget.reload(jsons[id], change, true);
                setTimeout(widget.render, 100);
            }
        });
    });
});

$(".fullscreen").click(function () {
    var wid = $(this).parents('section').attr('id');

    widgets[wid].then(function (widget) {
        widget.setFullscreen(!widget.getFullscreen());
    });
});
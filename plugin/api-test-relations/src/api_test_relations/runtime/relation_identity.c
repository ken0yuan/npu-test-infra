#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <internal/pycore_frame.h>
#include <descrobject.h>

#if PY_MAJOR_VERSION != 3 || PY_MINOR_VERSION != 11
#error "This identity adapter is for CPython 3.11; add a version adapter before using another interpreter."
#endif

/* Enter Python observer code with a thread-local compiler evaluation guard.
 * The setter is supplied through a runtime capability, not a source offset.
 * It must be a C callable so its entry cannot itself be compiled.
 */
static int relation_profile(PyObject *state, PyFrameObject *frame, int event, PyObject *arg) {
    static const char *names[] = {"call", "exception", "line", "return", "c_call", "c_exception", "c_return", "opcode"};
    if (event < 0 || event > PyTrace_OPCODE) return 0;
    PyObject *callback=PyTuple_GET_ITEM(state,0), *guard=PyTuple_GET_ITEM(state,1), *previous=NULL;
    if (guard != Py_None) {
        previous=PyObject_CallOneArg(guard,Py_None);
        if (!previous) return -1;
    }
    PyObject *result=PyObject_CallFunction(callback,"OsO",(PyObject *)frame,names[event],arg ? arg : Py_None);
    PyObject *exc_type=NULL, *exc_value=NULL, *exc_tb=NULL;
    if (!result) PyErr_Fetch(&exc_type,&exc_value,&exc_tb);
    if (previous) {
        PyObject *restored=PyObject_CallOneArg(guard,previous);
        Py_DECREF(previous);
        if (!restored) {
            Py_XDECREF(result);Py_XDECREF(exc_type);Py_XDECREF(exc_value);Py_XDECREF(exc_tb);
            return -1;
        }
        Py_DECREF(restored);
    }
    if (!result) {PyErr_Restore(exc_type,exc_value,exc_tb);return -1;}
    Py_DECREF(result);return 0;
}

static PyObject *set_profile(PyObject *self, PyObject *args) {
    PyObject *callback,*guard;
    if (!PyArg_ParseTuple(args,"OO",&callback,&guard)) return NULL;
    if (!PyCallable_Check(callback) || (guard!=Py_None && !PyCFunction_Check(guard))) {
        PyErr_SetString(PyExc_TypeError,"Expected callback and a native evaluation guard");return NULL;
    }
    PyObject *state=PyTuple_Pack(2,callback,guard);
    if (!state) return NULL;
    PyEval_SetProfile(relation_profile,state);Py_DECREF(state);
    Py_RETURN_NONE;
}

static PyObject *current_profile(PyObject *self, PyObject *unused) {
    PyThreadState *state=PyThreadState_Get();
    if (!state->c_profileobj) Py_RETURN_NONE;
    if (state->c_profilefunc==relation_profile) return Py_NewRef(PyTuple_GET_ITEM(state->c_profileobj,0));
    return Py_NewRef(state->c_profileobj);
}

/* The actual executing function, NOT a guess based on shared bytecode. */
static PyObject *frame_function(PyObject *self, PyObject *arg) {
    if (!PyFrame_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "expected frame");
        return NULL;
    }
    PyFrameObject *frame = (PyFrameObject *)arg;
    if (!frame->f_frame || !frame->f_frame->f_func) Py_RETURN_NONE;
    return Py_NewRef((PyObject *)frame->f_frame->f_func);
}

/* Descriptor identity also survives storing a bound method before monkeypatching. */
static PyObject *method_definition(PyObject *self, PyObject *arg) {
    if (PyCFunction_Check(arg))
        return PyLong_FromVoidPtr(((PyCFunctionObject *)arg)->m_ml);
    if (Py_IS_TYPE(arg, &PyMethodDescr_Type) || Py_IS_TYPE(arg, &PyClassMethodDescr_Type))
        return PyLong_FromVoidPtr(((PyMethodDescrObject *)arg)->d_method);
    Py_RETURN_NONE;
}

static PyObject *namespace_version(PyObject *self, PyObject *arg) {
    PyObject *dict = NULL;
    if (PyModule_Check(arg)) dict = PyModule_GetDict(arg);
    else if (PyType_Check(arg)) dict = ((PyTypeObject *)arg)->tp_dict;
    if (dict && PyDict_Check(dict))
        return PyLong_FromUnsignedLongLong(((PyDictObject *)dict)->ma_version_tag);
    Py_RETURN_NONE;
}

static PyObject *namespaces_changed(PyObject *self, PyObject *args) {
    PyObject *objects, *versions;
    if (!PyArg_ParseTuple(args, "O!O!", &PyTuple_Type, &objects, &PyTuple_Type, &versions)) return NULL;
    if (PyTuple_GET_SIZE(objects) != PyTuple_GET_SIZE(versions)) {
        PyErr_SetString(PyExc_ValueError, "namespace/version length mismatch");return NULL;
    }
    for (Py_ssize_t i=0;i<PyTuple_GET_SIZE(objects);i++) {
        PyObject *object=PyTuple_GET_ITEM(objects,i), *version=PyTuple_GET_ITEM(versions,i), *dict=NULL;
        if (version == Py_None) continue;
        if (PyModule_Check(object)) dict=PyModule_GetDict(object);
        else if (PyType_Check(object)) dict=((PyTypeObject *)object)->tp_dict;
        if (!dict || ((PyDictObject *)dict)->ma_version_tag != PyLong_AsUnsignedLongLong(version)) Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

#include "relation_guard.h"

static PyMethodDef methods[] = {
    {"set_profile", set_profile, METH_VARARGS, "Install compiler-isolated observer trampoline."},
    {"current_profile", guard_current_profile, METH_NOARGS, "Current logical profiling callback."},
    {"install_guard", install_guard, METH_VARARGS, "Multiplex profiler with opcode guard."},
    {"remove_guard", remove_guard, METH_NOARGS, "Restore native profiling and tracing clients."},
    {"profile_client", profile_client, METH_NOARGS, "Current profiling client."},
    {"trace_client", trace_client, METH_NOARGS, "Current tracing client."},
    {"guard_intact", guard_intact, METH_NOARGS, "Whether both guard channels remain installed."},
    {"guard_available", guard_available, METH_NOARGS, "Whether the opcode guard can repair profiler changes."},
    {"frame_function", frame_function, METH_O, "Return exact executing Python function."},
    {"method_definition", method_definition, METH_O, "Return process-local method definition identity."},
    {"namespace_version", namespace_version, METH_O, "Namespace mutation counter for incremental resolution."},
    {"namespaces_changed", namespaces_changed, METH_VARARGS, "Check namespace mutations without Python-level scanning."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_relation_identity", NULL, -1, methods};
PyMODINIT_FUNC PyInit__relation_identity(void) { return PyModule_Create(&module); }

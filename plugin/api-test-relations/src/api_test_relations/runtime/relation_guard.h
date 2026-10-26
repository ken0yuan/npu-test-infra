/* CPython 3.11 profile multiplexing guarded at opcode boundaries.
 * Native profiler objects and callbacks are forwarded unchanged. No CPU assembly.
 * The trace channel is multiplexed, not replaced with a Python line observer.
 */
typedef struct {
    PyObject *observer;
    Py_tracefunc profile;
    PyObject *profile_obj;
    Py_tracefunc trace;
    PyObject *trace_obj;
    PyObject *frames;
    int active;
} RelationGuard;

static int guarded_profile(PyObject *, PyFrameObject *, int, PyObject *);
static int guarded_trace(PyObject *, PyFrameObject *, int, PyObject *);

static RelationGuard *guard_data(PyObject *capsule) {
    return PyCapsule_GetPointer(capsule, "api_relation_guard");
}

static void guard_destroy(PyObject *capsule) {
    RelationGuard *g=guard_data(capsule);
    if (!g) {PyErr_Clear();return;}
    Py_XDECREF(g->observer);Py_XDECREF(g->profile_obj);
    Py_XDECREF(g->trace_obj);Py_XDECREF(g->frames);PyMem_Free(g);
}

static PyObject *find_guard(void) {
    PyThreadState *ts=PyThreadState_Get();
    if (ts->c_tracefunc==guarded_trace) return ts->c_traceobj;
    if (ts->c_profilefunc==guarded_profile) return ts->c_profileobj;
    return NULL;
}

static int guard_frame(RelationGuard *g, PyFrameObject *frame) {
    if (!PyDict_GetItem(g->frames,(PyObject *)frame)) {
        if (PyDict_SetItem(g->frames,(PyObject *)frame,
                          frame->f_trace_opcodes ? Py_True : Py_False)<0) return -1;
    }
    frame->f_trace_opcodes=1;
    return 0;
}

static int guard_rearm(PyObject *capsule) {
    RelationGuard *g=guard_data(capsule);
    PyThreadState *ts=PyThreadState_Get();
    if (!g || !g->active) return 0;
    if (ts->c_profilefunc!=guarded_profile || ts->c_profileobj!=capsule) {
        PyObject *obj=Py_XNewRef(ts->c_profileobj);
        g->profile=ts->c_profilefunc;
        if (obj==PyTuple_GET_ITEM(g->observer,0)) {Py_CLEAR(obj);g->profile=NULL;}
        Py_XSETREF(g->profile_obj,obj);
        PyEval_SetProfile(guarded_profile,capsule);
        if (PyErr_Occurred()) return -1;
    }
    return 0;
}

static int guarded_profile(PyObject *capsule, PyFrameObject *frame, int event, PyObject *arg) {
    Py_INCREF(capsule);
    RelationGuard *g=guard_data(capsule);
    int result=0;
    if (g && g->active) {
        result=relation_profile(g->observer,frame,event,arg);
        if (!result && g->active && g->profile) {
            PyObject *client=Py_XNewRef(g->profile_obj);
            Py_tracefunc fn=g->profile;
            result=fn(client,frame,event,arg);
            Py_XDECREF(client);
        }
    }
    Py_DECREF(capsule);return result;
}

static int guarded_trace(PyObject *capsule, PyFrameObject *frame, int event, PyObject *arg) {
    Py_INCREF(capsule);
    RelationGuard *g=guard_data(capsule);
    int result=0;
    if (!g || !g->active) goto done;
    if (guard_frame(g,frame)<0 || guard_rearm(capsule)<0) {result=-1;goto done;}
    PyObject *opcode=PyDict_GetItem(g->frames,(PyObject *)frame);
    if (g->trace && (event!=PyTrace_OPCODE || opcode==Py_True)) {
        /* Restore the client's own flag while invoking its callback. */
        frame->f_trace_opcodes=(opcode==Py_True);
        result=g->trace(g->trace_obj,frame,event,arg);
        if (!result && g->active) result=PyDict_SetItem(g->frames,(PyObject *)frame,
                                         frame->f_trace_opcodes ? Py_True : Py_False);
        if (g->active) frame->f_trace_opcodes=1;
    }
    if (g->active && event==PyTrace_RETURN) {
        opcode=PyDict_GetItem(g->frames,(PyObject *)frame);
        frame->f_trace_opcodes=(opcode==Py_True);
        if (opcode && PyDict_DelItem(g->frames,(PyObject *)frame)<0) result=-1;
    }
done:
    Py_DECREF(capsule);return result;
}

static PyObject *install_guard(PyObject *self, PyObject *args) {
    PyObject *callback,*eval_guard;
    if (!PyArg_ParseTuple(args,"OO",&callback,&eval_guard)) return NULL;
    PyObject *existing=find_guard();
    if (existing) {
        if (guard_rearm(existing)<0) return NULL;
        Py_RETURN_NONE;
    }
    if (!PyCallable_Check(callback) || (eval_guard!=Py_None && !PyCFunction_Check(eval_guard))) {
        PyErr_SetString(PyExc_TypeError,"Invalid relation observer");return NULL;
    }
    RelationGuard *g=PyMem_Calloc(1,sizeof(RelationGuard));
    if (!g) return PyErr_NoMemory();
    PyObject *capsule=PyCapsule_New(g,"api_relation_guard",guard_destroy);
    if (!capsule) {PyMem_Free(g);return NULL;}
    PyThreadState *ts=PyThreadState_Get();
    g->observer=PyTuple_Pack(2,callback,eval_guard);g->frames=PyDict_New();
    g->profile=ts->c_profilefunc;g->profile_obj=Py_XNewRef(ts->c_profileobj);
    g->trace=ts->c_tracefunc;g->trace_obj=Py_XNewRef(ts->c_traceobj);
    if (!g->observer || !g->frames) {Py_DECREF(capsule);return NULL;}
    g->active=1;
    PyFrameObject *frame=PyThreadState_GetFrame(ts);
    while (frame) {
        if (guard_frame(g,frame)<0) {Py_DECREF(frame);Py_DECREF(capsule);return NULL;}
        PyFrameObject *back=PyFrame_GetBack(frame);Py_DECREF(frame);frame=back;
    }
    PyEval_SetTrace(guarded_trace,capsule);
    PyEval_SetProfile(guarded_profile,capsule);
    Py_DECREF(capsule);
    if (PyErr_Occurred()) return NULL;
    Py_RETURN_NONE;
}

static PyObject *remove_guard(PyObject *self, PyObject *unused) {
    PyObject *capsule=find_guard();
    if (!capsule) Py_RETURN_NONE;
    Py_INCREF(capsule);
    RelationGuard *g=guard_data(capsule);g->active=0;
    Py_ssize_t pos=0;PyObject *frame,*opcode;
    while (PyDict_Next(g->frames,&pos,&frame,&opcode))
        ((PyFrameObject *)frame)->f_trace_opcodes=(opcode==Py_True);
    PyDict_Clear(g->frames);
    PyThreadState *ts=PyThreadState_Get();
    if (ts->c_tracefunc==guarded_trace) PyEval_SetTrace(g->trace,g->trace_obj);
    if (ts->c_profilefunc==guarded_profile) PyEval_SetProfile(g->profile,g->profile_obj);
    Py_DECREF(capsule);
    if (PyErr_Occurred()) return NULL;
    Py_RETURN_NONE;
}

static PyObject *profile_client(PyObject *self, PyObject *unused) {
    PyThreadState *ts=PyThreadState_Get();
    PyObject *obj=ts->c_profileobj;
    if (ts->c_profilefunc==guarded_profile) obj=guard_data(obj)->profile_obj;
    return Py_NewRef(obj ? obj : Py_None);
}

static PyObject *trace_client(PyObject *self, PyObject *unused) {
    PyThreadState *ts=PyThreadState_Get();
    PyObject *obj=ts->c_traceobj;
    if (ts->c_tracefunc==guarded_trace) obj=guard_data(obj)->trace_obj;
    return Py_NewRef(obj ? obj : Py_None);
}

static PyObject *guard_current_profile(PyObject *self, PyObject *unused) {
    PyThreadState *ts=PyThreadState_Get();
    if (ts->c_profilefunc==guarded_profile)
        return Py_NewRef(PyTuple_GET_ITEM(guard_data(ts->c_profileobj)->observer,0));
    return current_profile(self,unused);
}

static PyObject *guard_intact(PyObject *self, PyObject *unused) {
    PyThreadState *ts=PyThreadState_Get();
    return PyBool_FromLong(ts->c_tracefunc==guarded_trace && ts->c_profilefunc==guarded_profile);
}

static PyObject *guard_available(PyObject *self, PyObject *unused) {
    return PyBool_FromLong(PyThreadState_Get()->c_tracefunc==guarded_trace);
}

#include "dispatch-common.h"

#include "dispatch-binary.h"
#include "dispatch-copy.h"
#include "dispatch-flash-attention.h"
#include "dispatch-gated-mul-mat-id.h"
#include "dispatch-gated-mul-mat.h"
#include "dispatch-gather-add.h"
#include "dispatch-get-rows.h"
#include "dispatch-glu.h"
#include "dispatch-hyper-connection.h"
#include "dispatch-mul-mat-id.h"
#include "dispatch-mul-mat.h"
#include "dispatch-reduction.h"
#include "dispatch-amax-scale.h"
#include "dispatch-rmsnorm.h"
#include "dispatch-rope-set-rows.h"
#include "dispatch-scale.h"
#include "dispatch-unary.h"

#include <limits>
#include <sstream>

namespace ggml::hrx {

std::string float_config_value(float value) {
    std::ostringstream out;
    out.precision(std::numeric_limits<float>::max_digits10);
    out << value;
    return out.str();
}

void register_common_dispatches(DispatchRegistryBuilder & registry) {
    register_binary_dispatch(registry);
    register_copy_dispatch(registry);
    register_flash_attention_dispatches(registry);
    register_gated_mul_mat_id_dispatches(registry);
    register_gated_mul_mat_dispatches(registry);
    register_gather_add_dispatch(registry);
    register_get_rows_dispatches(registry);
    register_glu_dispatches(registry);
    register_hyper_connection_dispatches(registry);
    register_mul_mat_id_dispatches(registry);
    register_mul_mat_dispatches(registry);
    register_reduction_dispatches(registry);
    register_amax_scale_dispatch(registry);
    register_rope_set_rows_dispatches(registry);
    register_scale_dispatch(registry);
    register_unary_dispatch(registry);
    register_rmsnorm_dispatches(registry);
}

}  // namespace ggml::hrx

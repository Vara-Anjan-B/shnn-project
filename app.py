import streamlit as st
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns

from build import (
    apply_multi_layer_damage, create_base_model, get_acc,
    show_model_architecture,
    show_multi_layer_damage_circles,
    stability_validation,
    dual_path_healing,
    graph_based_layer_priority,
    multi_scale_localization,
    show_multi_scale_localization,
    get_cnn_layer_outputs,
    detect_cnn_damage,
    build_cnn_patches,
    extract_layers_weights,
    train_multi_layer_patches,
    detect_multi_layer_damage,
    build_multi_layer_patches,
    integrate_multi_layer_patches,
    evaluate_multi_layer_recovery,
    show_vulnerability_flow,
    show_multi_layer_patch_circles_for_struc,
    evaluate_patch_importance,
    compute_confidence_map,
    recovery_curve,
    apply_structural_damage,
    get_layer_outputs,
    compare_saved_outputs,
    find_damaged_layer,
    train_healing_patch,
    fgsm_attack, pgd_attack,
    find_damaged_layer,
    train_healed_model,
    show_layer_damage_circles,
    get_damaged_layer,
    build_patch,
    integrate_patch,
    freeze_except_patch,
    show_patch_layer_replacement,
    show_patch_layer_replacement_struc,
    show_layer_damage_circles_for_struc,
    show_layer_patch_circles_for_struc,
    evaluate_trust_fusion,
    show_shnn_pipeline,
    memory_update,
    create_cnn_model,
    air_multi_layer_attack,
    air_healing,
    detect_air_damage,
    air_multi_layer_healing
)

def adversarial_heal_pipeline(
    model,
    attack_type,
    X_train,
    y_train,
    X_test,
    y_test
):

    damaged_layer, diffs, names = get_damaged_layer(
        model,
        X_test,
        y_test,
        attack_type
    )

    output_dim = model.get_layer(damaged_layer).output.shape[1]

    patch = build_patch(output_dim)

    healed_model = integrate_patch(model, damaged_layer, patch)

    freeze_except_patch(healed_model, patch)

    # allow output to adapt
    for layer in healed_model.layers:
        if "output" in layer.name:
            layer.trainable = True

    healed_model = train_healed_model(
        healed_model,
        model,
        X_train,
        y_train,
        attack_type
    )

    return healed_model, damaged_layer, diffs, names

def air_heal_pipeline(
    model,
    layers,
    X_train,
    y_train
):

    healed = air_multi_layer_healing(
        model,
        layers,
        X_train,
        y_train
    )

    return healed

st.set_page_config(page_title="Simulate SHNN", layout="wide")
st.title("🧠 Simulate Self-Healing Neural Network")

uploaded_file = st.file_uploader("📂 Upload MNIST .mat file (e.g., mnist-original.mat)", type=["mat"])

def plot_bar(metrics, title,filename):
    fig, ax = plt.subplots()
    bars = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis", ax=ax)
    
    for bar, value in zip(bars.patches, metrics.values()):
        ax.annotate(f'{value:.1f}%', 
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black')
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)

    fig.savefig(f"images/{filename}.png", dpi=500, bbox_inches="tight")
    st.pyplot(fig)

def plot_weight_histograms(original_weights, damaged_weights, healed_weights,file_name, bins=100):
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    def flatten(weights_list):
        return np.concatenate([
            w.flatten() for w in weights_list
            if isinstance(w, np.ndarray)
        ])

    orig = flatten(original_weights)
    damaged = flatten(damaged_weights)
    healed = flatten(healed_weights)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(orig, bins=bins, histtype='step', linewidth=2, label='Original')
    ax.hist(damaged, bins=bins, histtype='step', linewidth=2, label='Damaged')
    ax.hist(healed, bins=bins, histtype='step', linewidth=2, label='Healed')
    ax.set_title("Weight Distribution Histogram")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f"images/{file_name}.png", dpi=500, bbox_inches="tight")
    st.pyplot(fig)

def prepare_image(img):
    img = np.squeeze(img)
    return img.reshape(28, 28) 
    
if uploaded_file:
    st.success("✅ File uploaded!")

    try:
        if 'data' not in st.session_state:
            mnist = loadmat(uploaded_file)
            mnist_data = mnist["data"].T
            mnist_label = mnist["label"][0]
            data = mnist_data / 255.0

            X_train, X_test, y_train, y_test = train_test_split(data, mnist_label, test_size=0.2, random_state=42)
            y_train_cat = to_categorical(y_train,len(np.unique(mnist_label)))
            y_test_cat = to_categorical(y_test,len(np.unique(mnist_label)))

            st.session_state.update({
                "X_train": X_train,
                "X_test": X_test,
                "y_train_cat": y_train_cat,
                "y_test_cat": y_test_cat,
                "data": data,
                "y": mnist_label,
                "model_created": False
            })

        st.subheader("📊 Dataset Overview")
        st.write(f"Data shape: `{st.session_state.data.shape}`")
        st.write(f"Label shape: `{st.session_state.y.shape}`")
        st.write(f"Unique classes: {np.unique(st.session_state.y)}")

        model_type = st.radio(
            "Select Model Type",
            ["MLP","CNN"],
            horizontal=True
        )
        

        if st.button("🚀 Create Base Model"):
            if model_type == "MLP":
                model = create_base_model(784,len(np.unique(st.session_state.y)),[256,128,64,32])
            else:
                model = create_cnn_model(len(np.unique(st.session_state.y)))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            with st.spinner("Training base model (10 epochs)..."):
                history=model.fit(st.session_state.X_train, st.session_state.y_train_cat, epochs=10, batch_size=128, verbose=0)
                loss, acc = model.evaluate(st.session_state.X_test, st.session_state.y_test_cat, verbose=0)
            st.subheader("Training Progress")

            # ---- Training Accuracy ----
            fig_acc, ax1 = plt.subplots(figsize=(6, 5))

            acc_key = 'accuracy'
            if acc_key not in history.history:
                acc_key = 'acc'
            train_acc = history.history[acc_key][-1]

            ax1.plot(history.history[acc_key], label='Training Accuracy', color='tab:blue')
            ax1.set_title('Training Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)

            # Save Accuracy plot
            fig_acc.savefig("images/Base_model_train_accuracy.png", dpi=500, bbox_inches="tight")
            st.pyplot(fig_acc)


            # ---- Training Loss ----
            fig_loss, ax2 = plt.subplots(figsize=(6, 5))

            ax2.plot(history.history['loss'], label='Training Loss', color='tab:orange')
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)

            # Save Loss plot
            fig_loss.savefig("images/Base_model_train_loss.png", dpi=500, bbox_inches="tight")
            st.pyplot(fig_loss)

            if model_type == "MLP":
                ref = get_layer_outputs(model, st.session_state.X_test[:10])
            else:
                ref,_ = get_cnn_layer_outputs(model, st.session_state.X_test[:10])

            st.session_state.update({
                "model": model,
                "model_created": True,
                "ref_outputs": ref,
                "X_sample_ref": st.session_state.X_test[:10]
            })
            st.session_state.original_model = tf.keras.models.clone_model(model)
            st.session_state.original_model.set_weights(model.get_weights())
            st.session_state.model_before_heal = tf.keras.models.clone_model(model)
            st.session_state.model_before_heal.set_weights(model.get_weights())
            show_model_architecture(model, st)
            st.success(f"✅ Base Model Trained - Train Accuracy: **{train_acc:.2%}** Test Accuracy: **{acc:.2%}**")

        if st.session_state.get("model_created"):
            st.header("⚔️ Choose Repair Path")
            path = st.radio("Select repair path:", ["Adversarial", "Structural"], horizontal=True)

            # ----------------------------- Adversarial Path -----------------------------
            if path == "Adversarial":
                st.subheader("🔐 Adversarial Repair")
                model = st.session_state.model
                X_sample = st.session_state.X_test[:100]
                y_sample = st.session_state.y_test_cat[:100]
                attack_type = st.selectbox("⚔️ Choose Adversarial Attack Type", ["FGSM", "PGD", "AIR-FGSM", "AIR-PGD"])
                if attack_type == "FGSM":
                    X_adv = fgsm_attack(model, X_sample, y_sample)

                elif attack_type == "PGD":
                    X_adv = pgd_attack(model, X_sample, y_sample)

                elif attack_type == "AIR-FGSM":
                    X_adv, layers, diffs, names = air_multi_layer_attack(
                        model,
                        X_sample,
                        y_sample,
                        "FGSM"
                    )
                    st.session_state.air_layers = layers
                    st.session_state.air_diffs = diffs
                    st.session_state.air_names = names

                elif attack_type == "AIR-PGD":
                    X_adv, layers, diffs, names = air_multi_layer_attack(
                        model,
                        X_sample,
                        y_sample,
                        "PGD"
                    )
                    st.session_state.air_layers = layers
                    st.session_state.air_diffs = diffs
                    st.session_state.air_names = names

                if st.button("🔒 Evaluate Model on Adversarial Samples"):

                    preds = {
                        "Normal": model.predict(X_sample),
                        attack_type: model.predict(X_adv)
                    }

                    accs = {k: np.mean(np.argmax(v, axis=1) == np.argmax(y_sample, axis=1)) * 100 for k, v in preds.items()}
                    
                    st.session_state.update({
                        "X_sample": X_sample,
                        "y_sample": y_sample,
                        "X_adv": X_adv,
                        "attack_type": attack_type
                    })

                    plot_bar(accs, "Accuracy Before Adversarial Healing",f"Normalvs{attack_type}_Damaged")
                    st.markdown("### 🔍 Sample Predictions (Clean vs Adversarial)")

                    num_to_show = 5
                    class_names = [str(i) for i in range(len(np.unique(st.session_state.y)))]

                    for i in range(num_to_show):
                        col1, col2 = st.columns(2)

                        true_label = class_names[np.argmax(y_sample[i])]
                        clean_pred = class_names[np.argmax(preds["Normal"][i])]
                        adv_pred = class_names[np.argmax(preds[attack_type][i])]

                        with col1:
                            st.markdown(f"**Clean Sample #{i+1}**")
                            st.image(prepare_image(X_sample[i]), width=50,
                                caption=f"True: {true_label} | Pred: {clean_pred}", use_container_width=True)

                        with col2:
                            st.markdown(f"**Adversarial Sample #{i+1}**")
                            st.image(prepare_image(X_adv[i]), width=50,
                                caption=f"True: {true_label} | Pred: {adv_pred}", use_container_width=True)
                    X_test = st.session_state.X_test
                    y_test_cat = st.session_state.y_test_cat
                    if "AIR" in attack_type:

                        show_multi_layer_damage_circles(
                            st.session_state.air_diffs,
                            st.session_state.air_names,
                            st.session_state.air_layers,
                            f"layer_damage_visualization_{attack_type}",
                            st
                        )

                    else:

                        a,b,c = get_damaged_layer(
                            model,
                            X_test,
                            y_test_cat,
                            attack_type
                        )

                        show_layer_damage_circles(
                            b,
                            c,
                            a,
                            f"layer_damage_visualization_{attack_type}",
                            st
                        )
                    st.session_state.model_before_heal = tf.keras.models.clone_model(model)
                    st.session_state.model_before_heal.set_weights(model.get_weights())
                if st.button("🛡️ Heal Model with Adversarial Training"):
                    with st.spinner("Healing using adversarial examples..."):

                        model = st.session_state.model
                        X_train = st.session_state.X_train
                        y_train_cat = st.session_state.y_train_cat
                        X_test = st.session_state.X_test
                        y_test_cat = st.session_state.y_test_cat

                        # ---------- AIR ----------
                        if "AIR" in attack_type:

                            layers = st.session_state.air_layers

                            healed_model = air_multi_layer_healing(
                                model,
                                layers,
                                X_train,
                                y_train_cat
                            )

                            # -------- generate adversarial data (same as FGSM) --------
                            if "FGSM" in attack_type:
                                X_adv = fgsm_attack(model, X_train[:15000], y_train_cat[:15000])
                            else:
                                X_adv = pgd_attack(model, X_train[:15000], y_train_cat[:15000])

                            X_total = np.concatenate([X_train[:15000], X_adv])
                            y_total = np.concatenate([y_train_cat[:15000], y_train_cat[:15000]])

                            # -------- train only patches --------
                            for layer in healed_model.layers:
                                layer.trainable = False

                            for layer in healed_model.layers:
                                if "meta_patch" in layer.name:
                                    layer.trainable = True

                            healed_model.compile(
                                optimizer="adam",
                                loss="categorical_crossentropy",
                                metrics=["accuracy"]
                            )

                            healed_model.fit(
                                X_total,
                                y_total,
                                epochs=8,
                                batch_size=128,
                                verbose=1
                            )

                            st.session_state.model = healed_model

                            show_multi_layer_patch_circles_for_struc(
                                [l.name for l in healed_model.layers],
                                layers,
                                f"layer_patch_visualization_{attack_type}",
                                st
                            )

                        # ---------- ORIGINAL CODE ----------
                        else:

                            damaged_layer,_,__ = get_damaged_layer(model, X_test,y_test_cat,attack_type)
                            output_dim = model.get_layer(damaged_layer).output.shape[1]

                            patch = build_patch(output_dim)

                            healed_model = integrate_patch(model, damaged_layer, patch)

                            freeze_except_patch(healed_model, patch)

                            healed_model = train_healed_model(
                                healed_model,
                                model,
                                X_train,
                                y_train_cat,
                                attack_type
                            )

                            st.session_state.model = healed_model

                            a,b,c = get_damaged_layer(
                                healed_model,
                                X_test,
                                y_test_cat,
                                attack_type
                            )

                            # auto patch name
                            patch_layer = [x for x in c if "patch" in x][0]

                            show_patch_layer_replacement(
                                c,
                                b,
                                patch_layer,
                                f"layer_patch_visualization_{attack_type}",
                                st
                            )

                        st.success("✅ Model healed using adversarial training")
                if st.button("📈 Re-evaluate After Adversarial Repair"):
                    model = st.session_state.model
                    X_sample = st.session_state.X_sample
                    y_sample = st.session_state.y_sample
                    preds = {
                        "Normal": model.predict(X_sample),
                        attack_type: model.predict(st.session_state.X_adv)
                    }
                    accs = {k: np.mean(np.argmax(v, axis=1) == np.argmax(y_sample, axis=1)) * 100 for k, v in preds.items()}
                    plot_bar(accs, "Accuracy After Adversarial Healing",f"Normalvs{attack_type}_Healed")
                    num_to_show = 10
                    class_names = [str(i) for i in range(len(np.unique(st.session_state.y)))]

                    for i in range(num_to_show):
                        col1, col2 = st.columns(2)

                        true_label = class_names[np.argmax(y_sample[i])]
                        clean_pred = class_names[np.argmax(preds["Normal"][i])]
                        adv_pred = class_names[np.argmax(preds[attack_type][i])]

                        with col1:
                            st.markdown(f"**Clean Sample #{i+1}**")
                            st.image(prepare_image(X_sample[i]), width=50,
                                caption=f"True: {true_label} | Pred: {clean_pred}", use_container_width=True)

                        with col2:
                            st.markdown(f"**Adversarial Sample #{i+1}**")
                            st.image(prepare_image(X_adv[i]), width=50,
                                caption=f"True: {true_label} | Pred: {adv_pred}", use_container_width=True)

            # ----------------------------- Structural Path -----------------------------
            elif path == "Structural":
                st.subheader("🧱 Structural Damage Repair")
                
                if st.button("💥 Apply Random Structural Damage"):
                    st.session_state.model, layer, st.session_state.mode,original_weights,damaged_weights = apply_structural_damage(st.session_state.model)
                    st.session_state.damage_type = "single"
                    st.session_state.damaged_layer = layer
                    st.session_state.org_weight = original_weights
                    st.session_state.dmg_weight = damaged_weights
                    st.error(f"💔 Structural damage applied to `{layer}` using `{st.session_state.mode}`")

                if st.button("💥 Apply Multi-Layer Damage"):

                    model = st.session_state.model

                    model, layers, modes, org_w, dmg_w = apply_multi_layer_damage(
                        model,
                        num_layers=2
                    )
                    st.session_state.model_before_heal = tf.keras.models.clone_model(model)
                    st.session_state.model_before_heal.set_weights(model.get_weights())

                    st.session_state.model = model
                    st.session_state.damage_type = "multi"
                    st.session_state.multi_damaged_layers = layers
                    st.session_state.org_weight = org_w
                    st.session_state.dmg_weight = dmg_w

                    st.error(f"Multi-layer damage applied: {layers}")

                if st.session_state.get("damage_type") == "single":
                    if st.button("🩻 Detect Damaged Layers"):
                        curr_outputs = get_layer_outputs(st.session_state.model, st.session_state.X_test[:10])
                        diffs = compare_saved_outputs(st.session_state.ref_outputs, curr_outputs)
                        
                        layer_names = [layer.name for layer in st.session_state.model.layers if isinstance(layer, tf.keras.layers.Dense)]

                        
                        layer = find_damaged_layer(diffs, layer_names)
                        show_layer_damage_circles_for_struc(diffs,layer_names,layer,f"layer_damage_visualization_{st.session_state.mode}",st)
                        st.session_state.damaged_layer = layer
                        st.warning(f"🔍 Most likely damaged layer: `{layer}`")

                    if st.button("📉 Test Accuracy After Damage"):
                        acc = get_acc(st.session_state.model, st.session_state.X_test, st.session_state.y_test_cat)
                        st.metric("🎯 Accuracy After Damage", f"{acc:.2f}%")
                        st.progress(acc / 100)

                    if st.button("🩹 Heal the Model"):
                        with st.spinner("Healing structurally damaged model..."):
                            healed_model, _, history,healed_weights = train_healing_patch(
                                st.session_state.model,
                                st.session_state.damaged_layer,
                                st.session_state.X_train,
                                st.session_state.y_train_cat
                            )
                            plot_weight_histograms(st.session_state.org_weight,st.session_state.dmg_weight,healed_weights,f"Histogram_{st.session_state.mode}")
                            st.session_state.model = healed_model
                            show_patch_layer_replacement_struc([layer.name for layer in healed_model.layers],'patch',f"layer_patch_visualization_{st.session_state.mode}",st)
                            st.success("✅ Model healing complete")
                            fig, ax = plt.subplots()
                            ax.plot(history.history['accuracy'], label='Training Accuracy')
                            ax.set_title("📈 Patch Training Accuracy (Structural Healing)")
                            ax.set_xlabel("Epoch")
                            ax.set_ylabel("Accuracy")
                            ax.legend()
                            fig.savefig(f"images/patch_training_acc_{st.session_state.mode}.png", dpi=500, bbox_inches="tight")
                            st.pyplot(fig)

                    if st.button("📊 Calculate Test Accuracy"):
                        acc = get_acc(st.session_state.model, st.session_state.X_test, st.session_state.y_test_cat)
                        
                        layer_names = [layer.name for layer in st.session_state.model.layers if isinstance(layer, tf.keras.layers.Dense)]
                        healed_op = get_layer_outputs(st.session_state.model, st.session_state.X_test[:10])
                        diffs = compare_saved_outputs(st.session_state.ref_outputs, healed_op)

                        show_layer_patch_circles_for_struc(diffs,layer_names,'patch',f"layer_patch_visualization_{st.session_state.mode}",st)
                        st.metric("🎯 Final Test Accuracy", f"{acc:.2f}%")
                        st.progress(acc / 100)

                if st.session_state.get("damage_type") == "multi":

                    if st.button("🩻 Detect Multi-Layer Damage"):

                        if model_type == "MLP":

                            layers, scores, diffs, layer_names = detect_multi_layer_damage(
                                st.session_state.model,
                                st.session_state.ref_outputs,
                                st.session_state.X_test[:10],
                                k=2
                            )
                            st.session_state.multi_damage_scores = scores

                        else:

                            layers, diffs, layer_names = detect_cnn_damage(
                                st.session_state.model,
                                st.session_state.ref_outputs,
                                st.session_state.X_test[:10]
                            )
                            st.session_state.multi_damage_scores = diffs

                        st.session_state.multi_damaged_layers = layers
                        show_multi_layer_damage_circles(
                            diffs,
                            layer_names,
                            layers,
                            "multilayer_damage",
                            st
                        )
                        show_vulnerability_flow(
                            layer_names,
                            diffs,
                            "vulnerability_flow",
                            st
                        )
                        layer_scores, neuron_scores = multi_scale_localization(
                            st.session_state.model,
                            st.session_state.X_test[:10],
                            st.session_state.ref_outputs
                        )

                        show_multi_scale_localization(
                            layer_scores,
                            neuron_scores,
                            layer_names,
                            st
                        )

                        st.warning(f"Top damaged layers: {layers}")

                    if st.session_state.get("damage_type") == "multi":

                        if st.button("🧩 Generate Multi-Layer Patches"):

                            if model_type == "MLP":
                                layer_names = [
                                    layer.name 
                                    for layer in st.session_state.model.layers
                                    if isinstance(layer, tf.keras.layers.Dense)
                                    and layer.name != "output"
                                ]
                                prioritized_layers, _ = graph_based_layer_priority(
                                    layer_names,
                                    st.session_state.multi_damage_scores
                                )

                                st.session_state.multi_damaged_layers = prioritized_layers[:2]

                                patches = build_multi_layer_patches(
                                    st.session_state.model,
                                    st.session_state.multi_damaged_layers
                                )
                            else:
                                patches = build_cnn_patches(
                                    st.session_state.model,
                                    st.session_state.multi_damaged_layers
                                )

                            st.session_state.multi_patches = patches

                            st.success(
                                f"Patches created for: {list(patches.keys())}"
                            )

                    if st.session_state.get("damage_type") == "multi":

                        if st.button("🩹 Heal Multi-Layer"):

                            healed_model = integrate_multi_layer_patches(
                                st.session_state.model,
                                st.session_state.multi_patches
                            )

                            # TRAIN PATCHES
                            healed_model, history = train_multi_layer_patches(
                                healed_model,
                                st.session_state.X_train,
                                st.session_state.y_train_cat
                            )

                            st.session_state.model = healed_model

                            layer_names = [layer.name for layer in healed_model.layers]

                            show_multi_layer_patch_circles_for_struc(
                                layer_names,
                                st.session_state.multi_damaged_layers,
                                "multilayer_patch_visualization",
                                st
                            )
                            healed_weights = extract_layers_weights(
                                healed_model,
                                st.session_state.multi_damaged_layers
                            )

                            # only plot if weights exist (MLP works, CNN skips)
                            if len(healed_weights) > 0:
                                plot_weight_histograms(
                                    st.session_state.org_weight,
                                    st.session_state.dmg_weight,
                                    healed_weights,
                                    "multilayer_histogram"
                                )
                            else:
                                st.info("CNN patch histogram skipped (no direct weights)")
                            results = evaluate_multi_layer_recovery(
                                st.session_state.original_model,
                                st.session_state.model_before_heal,
                                healed_model,
                                st.session_state.X_test,
                                st.session_state.y_test_cat
                            )

                            st.subheader("📊 Multi-Layer Healing Results")
                            
                            st.write(results)

                            st.success(
                                f"Multi-layer patches trained: {st.session_state.multi_damaged_layers}"
                            )

                if st.button("⚡ Dual Path Healing") and st.session_state.get("multi_damaged_layers"):

                    healed = dual_path_healing(
                        st.session_state.model,
                        st.session_state.multi_damaged_layers,
                        st.session_state.X_train,
                        st.session_state.y_train_cat,
                        "FGSM"
                    )

                    st.session_state.model = healed

                    st.success("Dual-path healing complete")
            if st.button("🧠 Trust-Aware Fusion"):

                base = get_acc(
                    st.session_state.original_model,
                    st.session_state.X_test,
                    st.session_state.y_test_cat
                )

                healed = get_acc(
                    st.session_state.model,
                    st.session_state.X_test,
                    st.session_state.y_test_cat
                )

                fused = evaluate_trust_fusion(
                    st.session_state.original_model,
                    st.session_state.model,
                    st.session_state.X_test,
                    st.session_state.y_test_cat
                )

                metrics = {
                    "Baseline": base,
                    "Healed": healed,
                    "Fusion": fused
                }

                plot_bar(metrics, "Trust-Aware Fusion Comparison", "fusion_compare")

                st.success(f"Fusion Accuracy: {fused:.2f}%")

            if st.button("🧠 Memory Update"):

                updated_model, updated, base, heal = memory_update(
                    st.session_state.original_model,
                    st.session_state.model,
                    st.session_state.X_test,                            
                    st.session_state.y_test_cat
                )

                st.session_state.model = updated_model
                if updated:
                    st.success(
                        f"Memory Updated ✅  {base:.2f}% → {heal:.2f}%"
                    )
                else:
                    st.warning(
                        f"Memory Kept Original ⚠️  {base:.2f}% vs {heal:.2f}%"
                    )

              
            if st.button("🧪 Stability Validation"):
                results = stability_validation(
                    st.session_state.model,
                    st.session_state.X_test,
                    st.session_state.y_test_cat
                )

                st.subheader("Model Stability")

                fig, ax = plt.subplots()

                ax.plot(
                    results["noise"],
                    results["accuracy"],
                    marker="o"
                )

                ax.set_xlabel("Noise Level")
                ax.set_ylabel("Accuracy %")
                ax.set_title("Stability Under Noise")

                st.pyplot(fig)

                st.metric(
                    "Stability Score",
                    f"{results['stability']:.2f}"
                )
            
            if st.button("🧠 Show SHNN Pipeline"):
                show_shnn_pipeline("shnn_pipeline", st)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

